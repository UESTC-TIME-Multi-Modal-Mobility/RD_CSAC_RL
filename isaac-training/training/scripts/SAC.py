import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world
from model import ViT,ConvNet
from torchrl.modules.distributions import TanhNormal
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from copy import deepcopy
# from torch.cuda.amp import autocast, GradScaler


class SAC(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # Get action dimension from action_spec
        self.action_dim = action_spec.shape[-1]

        # self.scaler = GradScaler()
        # Feature extractor for LiDAR()
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        # feature_extractor_network = ViT().to(self.device)  # Use ViT as the feature extractor
        # feature_extractor_network = ConvNet().to(self.device)  # Use ViT as the feature extractor
        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        # Feature extractor
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        actor_extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
            scale_lb=cfg.network.scale_lb,
        ).to(self.device)

        actor_module = TensorDictSequential(
            self.feature_extractor,
            TensorDictModule(
                nn.Linear(256, self.action_dim * 2),  # 256 -> 6 (假设action_dim=3)
                in_keys=["_feature"],
                out_keys=["actor_params"]
            ),
            TensorDictModule(
                actor_extractor,
                in_keys=["actor_params"],
                out_keys=["loc", "scale"]
            ),
        ).to(self.device)

        self.actor = ProbabilisticActor(
            spec=action_spec,
            in_keys=["loc", "scale"],
            module=actor_module,
            distribution_class=TanhNormal,
            distribution_kwargs={"tanh_loc": False},
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
            out_keys=[("agents", "action")]  # 直接输出到目标动作范围
        ).to(self.device)

        # SAC 需要两个 Q 网络 - 使用CatTensors来连接feature和action
        self.qvalue1 = TensorDictSequential(
            CatTensors(["_feature", ("agents", "action")], "_feature_action", del_keys=False),
            TensorDictModule(
                nn.Sequential(
                    nn.Linear(256 + self.action_dim, 256),  # feature + action
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                ),
                in_keys=["_feature_action"],
                out_keys=["state_action_value_1"]
            )
        ).to(self.device)

        self.qvalue2 = TensorDictSequential(
            CatTensors(["_feature", ("agents", "action")], "_feature_action", del_keys=False),
            TensorDictModule(
                nn.Sequential(
                    nn.Linear(256 + self.action_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                ),
                in_keys=["_feature_action"],
                out_keys=["state_action_value_2"]
            )
        ).to(self.device)

        # Target Q 网络
        from copy import deepcopy
        self.qvalue1_target = deepcopy(self.qvalue1)
        self.qvalue2_target = deepcopy(self.qvalue2)

        # 温度参数 alpha（自适应熵调节）
        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
        # 使用detach来避免计算图问题
        self.alpha = self.log_alpha.exp().detach()
        
        # SAC参数
        self.gamma = getattr(cfg, 'gamma', 0.99)  # 折扣因子

        # Loss related
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.qvalue1_optim = torch.optim.Adam(self.qvalue1.parameters(), lr=cfg.critic.learning_rate)
        self.qvalue2_optim = torch.optim.Adam(self.qvalue2.parameters(), lr=cfg.critic.learning_rate)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_learning_rate if hasattr(cfg, 'alpha_learning_rate') else 1e-7)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)

        # 使用no_grad来避免计算图
        with torch.no_grad():
            self.__call__(dummy_input)

        # 确保所有参数都分离计算图
        self._detach_all_params()

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.qvalue1.apply(init_)
        self.qvalue2.apply(init_)

    def _detach_all_params(self):
        """确保所有参数都分离计算图，避免deepcopy问题"""
        with torch.no_grad():
            for module in [self.feature_extractor, self.actor, self.qvalue1, self.qvalue2]:
                for param in module.parameters():
                    if param.grad is not None:
                        param.grad = None
                    param.data = param.data.detach()
                for buffer in module.buffers():
                    buffer.data = buffer.data.detach()
            
            # 处理alpha参数
            if self.log_alpha.grad is not None:
                self.log_alpha.grad = None
            self.log_alpha.data = self.log_alpha.data.detach()
            self.alpha = self.log_alpha.exp().detach()

    def __call__(self, tensordict):
        tensordict = tensordict.to(self.device)
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        # Note: For SAC, we don't call Q-networks in forward pass like PPO's critic
        # Q-networks are only used during training

        # Cooridnate change: transform local to world
        # SAC风格：TanhNormal已经输出合适范围的action
        actions = tensordict["agents", "action"] * self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    def train(self, replay_buffer, batch_size=256, tau=0.005, alpha=0.2):
        """SAC training step"""
        infos = []
        update_steps = getattr(self.cfg, 'update_steps', 1)
        for _ in range(update_steps):
            # Sample from replay buffer
            batch = replay_buffer.sample(batch_size).to(self.device)
            for epoch in range(self.cfg.training_epoch_num):
                batch1 = make_batch(batch, self.cfg.num_minibatches)
                for minibatch in batch1:
                    info = self._update_sac(minibatch, tau, alpha)
                    infos.append(info)
        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update_sac(self, batch, tau=0.005, alpha=None):
        """Single SAC update step"""
        if alpha is None:
            alpha = self.alpha
            
        # Feature extraction for current states
        batch_with_features = self.feature_extractor(batch)
        
        # Get current actions and log probs from policy
        current_actions_td = self.actor(batch_with_features)
        current_actions = current_actions_td[("agents", "action")]
        current_log_probs = current_actions_td.get("sample_log_prob", None)
        
        # Extract data from batch
        rewards = batch["next", "agents", "reward"]
        dones = batch["next", "terminated"]
        
        # Update Q-functions
        with torch.no_grad():
            # Get next state features and actions
            next_batch = TensorDict({
                "agents": batch["next", "agents"]
            }, batch_size=batch["next"].batch_size, device=self.device)
            next_batch_with_features = self.feature_extractor(next_batch)
            next_actions_td = self.actor(next_batch_with_features)
            next_actions = next_actions_td[("agents", "action")]
            next_log_probs = next_actions_td.get("sample_log_prob", torch.zeros(batch.batch_size[0], device=self.device))
            
            
            # 构造Q网络的输入TensorDict
            next_q_input = TensorDict({
                "_feature": next_batch_with_features["_feature"],
                ("agents", "action"): next_actions,
            }, batch_size=batch.batch_size, device=self.device)
            
            # Compute target Q-values using target networks
            next_q1 = self.qvalue1_target(next_q_input)["state_action_value_1"]
            next_q2 = self.qvalue2_target(next_q_input)["state_action_value_2"]
            next_q = torch.min(next_q1, next_q2)
            
            # 处理log_probs维度
            if next_log_probs.dim() == 1:
                next_log_probs = next_log_probs.unsqueeze(-1)  # [batch, 1]
            
            target_q = rewards + self.gamma * (1 - dones.float()) * (next_q - alpha * next_log_probs)
        
        # Current Q-values - 分别计算避免共享计算图
        current_actions_batch = batch[("agents", "action")]
        
        # 为Q1计算单独的输入
        current_q1_input = TensorDict({
            "_feature": batch_with_features["_feature"].detach().clone(),  # detach并clone避免共享
            ("agents", "action"): current_actions_batch.detach().clone(),
        }, batch_size=batch.batch_size, device=self.device)
        
        # 为Q2计算单独的输入
        current_q2_input = TensorDict({
            "_feature": batch_with_features["_feature"].detach().clone(),
            ("agents", "action"): current_actions_batch.detach().clone(),
        }, batch_size=batch.batch_size, device=self.device)
        
        current_q1 = self.qvalue1(current_q1_input)["state_action_value_1"]
        current_q2 = self.qvalue2(current_q2_input)["state_action_value_2"]
        
        # Q-function losses - 分别计算
        target_q_detached = target_q.detach()
        q1_loss = F.mse_loss(current_q1, target_q_detached)
        q2_loss = F.mse_loss(current_q2, target_q_detached)
        
        # Update Q-functions separately
        self.qvalue1_optim.zero_grad()
        q1_loss.backward()
        self.qvalue1_optim.step()
        
        self.qvalue2_optim.zero_grad()
        q2_loss.backward()
        self.qvalue2_optim.step()
        
        # Update actor - 重新前向传播获得新的actions
        fresh_batch_with_features = self.feature_extractor(batch)
        fresh_actions_td = self.actor(fresh_batch_with_features)
        fresh_actions = fresh_actions_td[("agents", "action")]
        fresh_log_probs = fresh_actions_td.get("sample_log_prob", None)
        
            
        fresh_q_input = TensorDict({
            "_feature": fresh_batch_with_features["_feature"].detach(),  # detach feature避免重复梯度
            ("agents", "action"): fresh_actions,
        }, batch_size=batch.batch_size, device=self.device)
        
        q1_new = self.qvalue1(fresh_q_input)["state_action_value_1"]
        q2_new = self.qvalue2(fresh_q_input)["state_action_value_2"]
        q_new = torch.min(q1_new, q2_new)
        
        if fresh_log_probs is not None:
            if fresh_log_probs.dim() == 1:
                fresh_log_probs = fresh_log_probs.unsqueeze(-1)
            actor_loss = (alpha * fresh_log_probs - q_new).mean()
        else:
            actor_loss = -q_new.mean()
        
        # Update actor
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        
        # Update alpha (temperature parameter)
        if fresh_log_probs is not None:
            target_entropy = getattr(self.cfg, 'target_entropy', -self.action_dim)
            alpha_loss = -(self.log_alpha * (fresh_log_probs.detach() + target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            # 使用detach来避免计算图问题
            self.alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
        
        # Soft update target networks
        self._soft_update(self.qvalue1_target, self.qvalue1, tau)
        self._soft_update(self.qvalue2_target, self.qvalue2, tau)
        
        return TensorDict({
            "actor_loss": actor_loss,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "actor_grad_norm": actor_grad_norm,
        }, [])
    
    def _soft_update(self, target, source, tau):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)