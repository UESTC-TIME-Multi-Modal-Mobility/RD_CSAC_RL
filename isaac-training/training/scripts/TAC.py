import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world,GaussianActor
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from torchrl.modules.distributions import TanhNormal
from copy import deepcopy
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, device):
        super().__init__()
        # lidar特征提取

        feature_extractor_network = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.Conv2d(in_channels=4,out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.Conv2d(in_channels=16,out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(in_features=288,out_features=128), nn.LayerNorm(128),
        )
        # 动态障碍特征提取
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(50, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.LayerNorm(64),
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
                TensorDictModule(nn.Sequential(
                    nn.Linear(200, 256),
                    nn.LeakyReLU(),
                    nn.LayerNorm(256),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(),
                    nn.LayerNorm(256),
                ), in_keys=["_feature"], out_keys=["_feature_"]),
                TensorDictModule(GaussianActor(action_dim), in_keys=["_feature_"], out_keys=["loc", "scale"])
            ),
            in_keys=["loc", "scale"],
            out_keys=["action_normalized"], 
            distribution_class=TanhNormal,
            return_log_prob=True
        ).to(device)

    def forward(self, state):
        tensordict = TensorDict({"observation": state}, batch_size=state.shape[0])
        tensordict = self.feature_extractor(tensordict)
        tensordict = self.actor(tensordict)
        return tensordict["action_normalized"], tensordict["loc"],tensordict["scale"]

class CriticNetwork(nn.Module):
    def __init__(self,obs_dim ,action_dim, device):
        super().__init__()
        # lidar特征提取

        feature_extractor_network = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.Conv2d(in_channels=4,out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.Conv2d(in_channels=16,out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(288,128), nn.LayerNorm(128),
        )
        # 动态障碍特征提取
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(50, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.LayerNorm(64),
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
                nn.Sequential(
                    nn.Linear(200 + action_dim, 256),
                    nn.LeakyReLU(),
                    nn.LayerNorm(256),
                    nn.Linear(256, 256),
                    nn.LeakyReLU(),
                    nn.LayerNorm(256),
                    nn.Linear(256, 1),
                ),
                in_keys=["_feature_action"],       # <- 这里要读取拼接后的键
                out_keys=["state_action_value"],   # <- 直接输出最终值到 state_action_value
            ),
        ).to(device)
    def forward(self, s,a):
        tensordict = TensorDict({"observation": s, "action_normalized": a.squeeze(1)}, batch_size=s.shape[0])
        tensordict = self.feature_extractor(tensordict)
        q = self.qvalue(tensordict)["state_action_value"]
        return q
    



class SAC(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        # Initialize the SAC agent with configuration, observation and action specs, and device
        self.cfg = cfg
        self.obs_dim = observation_spec
        self.act_dim = action_spec
        self.q = cfg.entropic_index
        self.device = device
                # 兼容 action_spec 为整数或有 shape 的 Spec，提取 action_dim
        if hasattr(self.act_dim, "shape"):
            # action_spec.shape 可能是 tuple，例如 (action_dim,) 或 (n_agents, action_dim)
            shape = tuple(self.act_dim.shape)
            # 取最后一维作为 action_dim（如果是复合维度，请按需修改）
            self.act_dim = int(shape[-1]) if len(shape) > 0 else int(shape[0])
        else:
            self.act_dim = int(self.act_dim)
        # Initialize networks
        self.actor = ActorNetwork(self.obs_dim, self.act_dim, device).to(self.device)
        self.critic1 = CriticNetwork(self.obs_dim, self.act_dim, device).to(self.device)
        self.critic2 = CriticNetwork(self.obs_dim, self.act_dim, device).to(self.device)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        #Initialize Temperature parameter
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(5, device=device)))
        self.alpha = self.log_alpha.exp().detach()
        self.target_entropy = -float(self.act_dim)
        
        #Initialize Parameters
        self.gamma = getattr(cfg, 'gamma', 0.99)
        self.action_limit = getattr(cfg.actor, 'action_limit', 2.0)
        
        # Improved optimizers with different learning rates
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.critic.learning_rate)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.critic.learning_rate)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_learning_rate)


        def init_(module):
            from torch.nn.parameter import UninitializedParameter
            # 只对已初始化的常见线性/卷积层做正交初始化，跳过 lazy 未初始化参数
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                w = getattr(module, "weight", None)
                b = getattr(module, "bias", None)
                if w is None or isinstance(w, UninitializedParameter):
                    return
                nn.init.orthogonal_(module.weight, 0.01)
                if b is not None and not isinstance(b, UninitializedParameter):
                    nn.init.constant_(module.bias, 0.0)
        self.actor.apply(init_)
        self.critic1.apply(init_)
        self.critic2.apply(init_)
        self.critic1_target.apply(init_)
        self.critic2_target.apply(init_)
    def tsallis_entropy_log_q(self, x, q):
        safe_x = torch.max(x, torch.Tensor([1e-6]).to(self.device))

        if q == 1:
            log_q_x = torch.log(safe_x)
        else:
            log_q_x = (safe_x.pow(q-1)-1)/(q-1)
        return log_q_x.sum(dim=-1)
    
    def get_action(self, state, deterministic=True):
        if deterministic:
            with torch.no_grad():
                action, mu, log_std = self.actor(state)
            return action
        else:
            _ , mu, log_std = self.actor(state)
            std = log_std.exp().clamp(min=1e-6)
            normal = torch.distributions.Normal(mu, std)
            x_t = normal.rsample()
            actions = torch.tanh(x_t)

            # Calculate log probabilities with correction for tanh
            log_probs = normal.log_prob(x_t) - torch.log(1 - actions.pow(2) + 1e-6)
            # log_probs = log_probs.sum(-1)
            exp_log_probs = torch.exp(log_probs)
            log_probs = self.tsallis_entropy_log_q(exp_log_probs, self.q)
            return actions,log_probs
    def __call__(self,td):
        td = td.to(self.device)
        action_n, mu, log_std= self.actor(td["agents","observation"])
        actions_world = self.actions_to_world(action_n, td).squeeze(-1)
        td["agents","action"] = actions_world
        td["agents","action_normalized"] = action_n
        return td
    def train(self,replay_buffer, batch_size, tau=0.005):
        """SAC training step with improved stability"""
        train_tds = []
        # Sample batch
        for _ in range(self.cfg.num_minibatches):
            batch = replay_buffer.sample(batch_size).to(self.device)
            states = batch['agents','observation'].squeeze(-1).to(self.device)
            actions = batch['agents','action_normalized'].to(self.device)  # Normalize actions
            rewards = batch['next', 'agents','reward'].squeeze(1).to(self.device)
            next_states = batch['next', 'agents','observation'].squeeze(-1).to(self.device)
            # dones = (
            #     batch['next', 'terminated'].squeeze(-1).to(torch.bool)
            #     | batch['next', 'truncated'].squeeze(-1).to(torch.bool)
            # ).float().to(self.device)
            dones = batch['next', 'terminated'].squeeze(-1).to(torch.bool).float().to(self.device)
            # ============ Update Critics ============
            with torch.no_grad():
                # Sample next actions using current policy
                next_actions,next_log_probs = self.get_action(next_states,deterministic=False)
                # _ ,next_mu, next_log_std = self.actor(next_states)
                # next_std = next_log_std.exp().clamp(min=1e-6)
                # next_normal = torch.distributions.Normal(next_mu, next_std)
                # next_x_t = next_normal.rsample()
                # next_actions = torch.tanh(next_x_t)
                
                # # Calculate log probabilities with correction for tanh
                # next_log_probs = next_normal.log_prob(next_x_t) - torch.log(1 - next_actions.pow(2) + 1e-6)
                # next_log_probs = next_log_probs.sum(-1)
                
                # Target Q values
                next_q1 = self.critic1_target(next_states, next_actions).squeeze(-1)
                next_q2 = self.critic2_target(next_states, next_actions).squeeze(-1)
                next_q = torch.min(next_q1, next_q2)
                
                # Compute target with entropy regularization
                target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)
            
            # Current Q values
            q1 = self.critic1(states, actions).squeeze(-1)
            q2 = self.critic2(states, actions).squeeze(-1)
            
            # Critic losses
            critic1_loss = F.mse_loss(q1, target_q)
            critic2_loss = F.mse_loss(q2, target_q)
            
            # Update critics
            self.critic1_optim.zero_grad()
            critic1_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)  # Gradient clipping
            self.critic1_optim.step()
            
            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)  # Gradient clipping
            self.critic2_optim.step()
            
            # ============ Update Actor ============
            # Resample actions for actor update
            actions_new, log_probs = self.get_action(states,deterministic=False)
            # _ ,mu, log_std = self.actor(states)
            # std = log_std.exp().clamp(min=1e-6)
            # normal = torch.distributions.Normal(mu, std)
            # x_t = normal.rsample()
            # actions_new = torch.tanh(x_t)
            
            # # Calculate log probabilities
            # log_probs = normal.log_prob(x_t) - torch.log(1 - actions_new.pow(2) + 1e-6)
            # log_probs = log_probs.sum(-1)
            
            # Q values for new actions
            q1_new = self.critic1(states, actions_new).squeeze(-1)
            q2_new = self.critic2(states, actions_new).squeeze(-1)
            q_min = torch.min(q1_new, q2_new)
            
            # Actor loss
            actor_loss = (self.alpha * log_probs - q_min).mean()
            
            # Update actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Gradient clipping
            self.actor_optim.step()
            
            # ============ Update Temperature ============

            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            # Update temperature
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            
            # ============ Soft Update Target Networks ============
            self._soft_update(self.critic1_target, self.critic1, tau)
            self._soft_update(self.critic2_target, self.critic2, tau)
            

            train_td =  TensorDict({
                "actor_loss": actor_loss.item(),
                "q1_loss": critic1_loss.item(),
                "q2_loss": critic2_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": self.alpha.item(),
                "actor_lp": log_probs.mean(),
                "q1": q1.mean(),
                "q_min": q_min.mean(),
                "q1_new": q1_new.mean(),
                # "td_error": (q1_new - target_q).mean(),
                # "td_error_target": (q1 - target_q).mean(),
            }, [])
            train_tds.append(train_td)
        loss_infos = torch.stack(train_tds).to_tensordict()
        loss_infos = loss_infos.apply(torch.mean, batch_size=[])
        return {k: v.mean().item() for k, v in loss_infos.items()}
    def actions_to_world(self, actions, tensordict):
        """将动作从局部坐标系转换到世界坐标系"""
        actions = actions * self.cfg.actor.action_limit
        actions_world = vec_to_world(actions,tensordict["agents", "observation", "direction"])
        return actions_world
    def _soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)