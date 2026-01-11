import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world,GaussianActor
from model import ViT,ConvNet
from torchrl.modules.distributions import TanhNormal
from tensordict.nn.distributions import NormalParamExtractor
from copy import deepcopy
from tensordict.tensordict import TensorDict
# from torch.cuda.amp import autocast, GradScaler


class SAC(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.r_norm = ValueNorm(1).to(self.device)
        # Get action dimension from action_spec
        self.action_dim = action_spec.shape[-1]

        self.actor = ActorNetwork(self.action_dim,self.device).to(self.device)
        self.qvalue1 = CriticNetwork(self.action_dim,self.device).to(self.device)
        self.qvalue2 = CriticNetwork(self.action_dim,self.device).to(self.device)

        # Target Q 网络
        from copy import deepcopy
        self.qvalue1_target = deepcopy(self.qvalue1)
        self.qvalue2_target = deepcopy(self.qvalue2)

        # 温度参数 alpha（自适应熵调节）
        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
        # self.log_alpha = nn.Parameter(torch.tensor([0.693], device=device))
        self.alpha = self.log_alpha.exp().detach()
        
        # SAC参数
        self.gamma = getattr(cfg, 'gamma', 0.99)  # 折扣因子

        # Optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.qvalue1_optim = torch.optim.Adam(self.qvalue1.parameters(), lr=cfg.critic.learning_rate)
        self.qvalue2_optim = torch.optim.Adam(self.qvalue2.parameters(), lr=cfg.critic.learning_rate)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_learning_rate if hasattr(cfg, 'alpha_learning_rate') else 1e-4)

        # Dummy Input for nn lazy module
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)

        with torch.no_grad():
            self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.qvalue1.apply(init_)
        self.qvalue2.apply(init_)
        self.qvalue1_target.apply(init_)
        self.qvalue2_target.apply(init_)
        

    def __call__(self, tensordict):
        tensordict = tensordict.to(self.device)
        actor,actor_lp = self.actor(tensordict["agents"])
        tensordict["agents","action_normalized"] = actor["action_normalized"]
        self.qvalue1(TensorDict({"observation": tensordict["agents", "observation"],"action_normalized": tensordict["agents", "action_normalized"]},batch_size=tensordict.batch_size, device=self.device))
        self.qvalue2(TensorDict({"observation": tensordict["agents", "observation"],"action_normalized": tensordict["agents", "action_normalized"]},batch_size=tensordict.batch_size, device=self.device))
        self.qvalue1_target(TensorDict({"observation": tensordict["agents", "observation"],"action_normalized": tensordict["agents", "action_normalized"]},batch_size=tensordict.batch_size, device=self.device))
        self.qvalue2_target(TensorDict({"observation": tensordict["agents", "observation"],"action_normalized": tensordict["agents", "action_normalized"]}, batch_size=tensordict.batch_size, device=self.device))
        actions_world = self.actions_to_world(tensordict[("agents", "action_normalized")], tensordict)
        tensordict["agents", "action"] = actions_world
        return tensordict

    def train(self, replay_buffer, batch_size=256, tau=0.005, alpha=0.2):
        """SAC training step"""
        loss_infos = []
        reward_infos = []
        self.replay_buffer = replay_buffer
        update_steps = getattr(self.cfg, 'training_epoch_num', 5)
        for _ in range(update_steps):
            # Sample from replay buffer
            batch= replay_buffer.sample(batch_size).to(self.device)
            # with torch.no_grad():
            # self.r_norm.update(batch["next", "agents", "reward"])
            # normed_reward = self.r_norm.normalize(batch["next", "agents", "reward"])
            # batch["next", "agents", "reward"] = normed_reward
            batch1 = make_batch(batch, self.cfg.num_minibatches)
            for minibatch in batch1:
                loss_info,reward_info = self._update_sac(minibatch, tau, alpha)
                loss_infos.append(loss_info)
                
                # # 计算 TD-Error
                # td_error = self._compute_td_error(minibatch)
                # td_errors.append(td_error)
                # indices_list.append(indices)
        loss_infos = torch.stack(loss_infos).to_tensordict()
        loss_infos = loss_infos.apply(torch.mean, batch_size=[])
        return {k: v.mean().item() for k, v in loss_infos.items()}


    def _update_sac(self, batch, tau=0.005, alpha=None):
        """Single SAC update step"""
        #===================== 1. 公共特征提取 =====================
        # Extract data from batch
        rewards = batch["next", "agents", "reward"]
        # rewards_init = self.r_norm.denormalize(rewards)
        dones = (
            batch['next', 'terminated'].squeeze(-1).to(torch.bool)
            | batch['next', 'truncated'].squeeze(-1).to(torch.bool)
        ).float().to(self.device)
        observations = batch["agents"]
        next_observations = batch["next", "agents"]
        actors = batch["agents", "action_normalized"]
        # ===================== 2. Actor更新分支 =====================
        # 2.1 使用当前策略重新采样动作
        actions,actor_lp = self.actor(observations)
        # actor_lp = actions_td.get("sample_log_prob")
        next_actions, next_actions_lp = self.actor(next_observations)
        # print(f"Actor log_prob mean: {actor_lp.mean().item()}")
        q_input = TensorDict({
            "observation": observations["observation"],
            "action_normalized": actors.squeeze(1),
        }, batch_size=batch.batch_size, device=self.device)
        q1 = self.qvalue1(q_input)
        q2 = self.qvalue2(q_input)

        actor_q_input = TensorDict({
            "observation": observations["observation"],
            "action_normalized": actions["action_normalized"].squeeze(1),
        }, batch_size=batch.batch_size, device=self.device)
        next_q_input = TensorDict({
            "observation": next_observations["observation"],
            "action_normalized": next_actions["action_normalized"].squeeze(1),
        }, batch_size=batch.batch_size, device=self.device)
        
        q1_actor = self.qvalue1(actor_q_input)
        q2_actor = self.qvalue2(actor_q_input)
        q_min = torch.min(q1_actor, q2_actor)
        next_q1 = self.qvalue1_target(next_q_input)
        next_q2 = self.qvalue2_target(next_q_input)
        next_q_min = torch.min(next_q1, next_q2)
        
        # loss
        v_backup = next_q_min - self.alpha * next_actions_lp.unsqueeze(1)
        v_backup = v_backup.detach()
        q_backup = rewards + self.gamma * (1 - dones.float()) * v_backup
        actor_loss = (self.alpha * actor_lp - q_min).mean()
        q1_loss = F.mse_loss(q1, q_backup.detach())
        q2_loss = F.mse_loss(q2, q_backup.detach())
        # q1_loss = ((q1 - q_backup.detach()).pow(2).squeeze(-1) * weights).mean()
        # q2_loss = ((q2 - q_backup.detach()).pow(2).squeeze(-1) * weights).mean()
        # 计算并写回td_error
        # td_error = (q1.detach() - q_backup.detach()).abs().squeeze(-1) + 1e-6
        # batch.set("td_error", td_error)
        # # 更新优先级
        # self.replay_buffer.update_tensordict_priority(batch)
        
        # ===================== 4. 温度参数α更新 =====================
        if actor_lp is not None:
            target_entropy = getattr(self.cfg, 'target_entropy', -self.action_dim)
            alpha_loss = -(self.log_alpha * (actor_lp + target_entropy).detach()).mean()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
        
        # ===================== 5. 执行所有更新 =====================


        # 5.2 更新Q1网络
        self.qvalue1_optim.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qvalue1.parameters(), max_norm=5.0)
        q1_grad_norm = self.calc_total_grad_norm(self.qvalue1.parameters()) 
        self.qvalue1_optim.step()
        
        # 5.3 更新Q2网络
        self.qvalue2_optim.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qvalue2.parameters(), max_norm=5.0)
        q2_grad_norm = self.calc_total_grad_norm(self.qvalue2.parameters()) 
        self.qvalue2_optim.step()
        # 5.1 更新Actor和Feature Extractor
        self.actor_optim.zero_grad()
        actor_loss.backward()  
        actor_grad_norm = self.calc_total_grad_norm(self.actor.parameters()) 
        self.actor_optim.step()
        
        # 5.4 更新温度参数
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        
        # ===================== 6. 软更新目标网络 =====================
        self._soft_update(self.qvalue1_target, self.qvalue1, tau)
        self._soft_update(self.qvalue2_target, self.qvalue2, tau)
        
        return TensorDict({
            "actor_loss": actor_loss,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
        }, [])
    
    def _soft_update(self, target, source, tau):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
    
    def actions_to_world(self, actions, tensordict):
        """将动作从局部坐标系转换到世界坐标系"""
        actions = actions * self.cfg.actor.action_limit
        actions_world = vec_to_world(actions,tensordict["agents", "observation", "direction"])
        return actions_world
    # 再计算裁剪后的梯度范数
    def calc_total_grad_norm(self,parameters):
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    # def _compute_td_error(self, batch):
    #     with torch.no_grad():
    #         # 当前 Q 值
    #         current_actions = batch[("agents", "action_normalized")]
    #         current_q1_input = TensorDict({
    #             "agents": batch["agents"],
    #             ("agents", "action_normalized"): current_actions,
    #         }, batch_size=batch.batch_size, device=self.device)
    #         current_q1 = self.qvalue1(current_q1_input)["state_action_value"]

    #         # 目标 Q 值
    #         rewards = batch["next", "agents", "reward"]
    #         dones = batch["next", "terminated"]
    #         next_actions_td = self.actor(batch["next"])
    #         next_actions = next_actions_td[("agents", "action_normalized")]
    #         next_q_input = TensorDict({
    #             "agents": batch["next", "agents"],
    #             ("agents", "action_normalized"): next_actions,
    #         }, batch_size=batch.batch_size, device=self.device)
    #         next_q1 = self.qvalue1_target(next_q_input)["state_action_value"]
    #         next_q2 = self.qvalue2_target(next_q_input)["state_action_value"]
    #         next_q = torch.min(next_q1, next_q2)
    #         target_q = rewards + self.gamma * (1 - dones.float()) * next_q

    #         # 计算 TD-Error
    #         td_error = target_q - current_q1
    #     return td_error.abs()  # 返回绝对值作为优先级

class ActorNetwork(nn.Module):
    def __init__(self, action_dim, device):
        super().__init__()
        # lidar特征提取
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        )
        # 动态障碍特征提取
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        )
        # 总特征拼接+MLP
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            # TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
            TensorDictModule(nn.LayerNorm(200), ["_feature"], ["_feature"]),
        ).to(device)
        # 
        self.actor = ProbabilisticActor(
            TensorDictSequential(
                TensorDictModule(make_mlp([256, 256]), in_keys=["_feature"], out_keys=["_feature_"]),
                TensorDictModule(GaussianActor(action_dim), in_keys=["_feature_"], out_keys=["loc", "scale"])
            ),
            in_keys=["loc", "scale"],
            out_keys=["action_normalized"], 
            distribution_class=TanhNormal,
            return_log_prob=True
        ).to(device)

    def forward(self, tensordict):
        tensordict = self.feature_extractor(tensordict)
        tensordict = self.actor(tensordict)
        actor_lp = tensordict.get("sample_log_prob")
        return tensordict, actor_lp

class CriticNetwork(nn.Module):
    def __init__(self, action_dim, device):
        super().__init__()
        # lidar特征提取
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        )
        # 动态障碍特征提取
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        )
        # 总特征拼接+MLP
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            # TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
            TensorDictModule(nn.LayerNorm(200), ["_feature"], ["_feature"]),
        ).to(device)
        
        # Q网络
        self.qvalue = TensorDictSequential(
            CatTensors(["_feature", "action_normalized"], "_feature_action", del_keys=False),
            TensorDictModule(
                make_mlp([action_dim+256, action_dim+256, 1]),
                # nn.Linear(action_dim+256, 1),
                in_keys=["_feature_action"],
                out_keys=["state_action_value"]
            ),
        ).to(device)
    def forward(self, tensordict):
        tensordict = self.feature_extractor(tensordict)
        q = self.qvalue(tensordict)["state_action_value"]
        return q
