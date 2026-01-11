'''
Author: zdytim zdytim@foxmail.com
Date: 2026-01-06 11:20:29
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-06 12:40:13
FilePath: /NavRL/isaac-training/training/scripts/models/sac_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
SAC Model Manager
=================
抽象的 SAC 模型管理模块，将 SAC 模型的核心逻辑从训练脚本中分离出来。
提供统一的模型创建、加载、保存和配置管理接口。

主要功能：
1. SACFeatureExtractor: 基于 CNN 的特征提取器，用于 lidar 和动态障碍物
2. ActorNetwork: SAC Actor 网络，输出 TanhNormal 分布
3. CriticNetwork: SAC Critic 网络（Q函数）
4. SACModel: 完整的 SAC 模型，包含 Actor、双 Critic、Target Network 和 Temperature
5. SACModelManager: 模型管理器，提供统一的配置和状态管理接口

作者: NavRL Team
日期: 2026年1月6日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from torchrl.modules.distributions import TanhNormal
from copy import deepcopy
import os
import tempfile
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

# wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Model uploading will be disabled.")

# 导入项目依赖
from utils import vec_to_world, GaussianActor


class SACFeatureExtractor(nn.Module):
    """
    SAC 共享特征提取器
    
    功能：
    1. Lidar CNN 特征提取
    2. 动态障碍物编码器
    3. 状态编码器
    4. 特征融合（拼接 + LayerNorm）
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        # Lidar 特征提取网络
        self.lidar_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=[5, 3], padding=[2, 1]), 
            nn.ELU(), 
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), 
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), 
            nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(in_features=288, out_features=128), 
            nn.LayerNorm(128),
        ).to(device)
        
        # 动态障碍物特征提取网络
        self.dyn_obs_net = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(50, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.LayerNorm(64),
        ).to(device)
        
        # TensorDict 格式的特征提取流水线
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(self.lidar_cnn, [("observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(self.dyn_obs_net, [("observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            TensorDictModule(nn.LayerNorm(200), ["_feature"], ["_feature"]),
        ).to(device)
        
        print(f"✅ SACFeatureExtractor initialized on {device}")
    
    def forward(self, observation: Dict) -> torch.Tensor:
        """
        前向传播
        
        Args:
            observation: 观测字典，包含 lidar, dynamic_obstacle, state
            
        Returns:
            feature: 融合特征 [Batch, 200]
        """
        batch_size = observation["lidar"].shape[0]
        tensordict = TensorDict({"observation": observation}, batch_size=batch_size)
        tensordict = self.feature_extractor(tensordict)
        return tensordict["_feature"]


class ActorNetwork(nn.Module):
    """
    SAC Actor 网络
    
    输出 TanhNormal 分布的参数（loc, scale）
    """
    
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        
        # 特征提取器
        self.feature_extractor = SACFeatureExtractor(device)
        
        # Actor head
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
        
        print(f"✅ ActorNetwork initialized with action_dim={action_dim}")

    def forward(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 观测字典
            
        Returns:
            action_normalized: 归一化动作 [Batch, action_dim]
            loc: 均值 [Batch, action_dim]
            scale: 标准差 [Batch, action_dim]
        """
        tensordict = TensorDict({"observation": state}, batch_size=state["lidar"].shape[0])
        tensordict = self.feature_extractor.feature_extractor(tensordict)
        tensordict = self.actor(tensordict)
        return tensordict["action_normalized"], tensordict["loc"], tensordict["scale"]


class CriticNetwork(nn.Module):
    """
    SAC Critic 网络（Q 函数）
    
    输入状态和动作，输出 Q 值
    """
    
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        
        # 特征提取器
        self.feature_extractor = SACFeatureExtractor(device)
        
        # Q 网络
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
                in_keys=["_feature_action"],
                out_keys=["state_action_value"],
            ),
        ).to(device)
        
        print(f"✅ CriticNetwork initialized with action_dim={action_dim}")
    
    def forward(self, s: Dict, a: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            s: 观测字典
            a: 动作 [Batch, action_dim]
            
        Returns:
            q: Q 值 [Batch, 1]
        """
        tensordict = TensorDict(
            {"observation": s, "action_normalized": a.squeeze(1)}, 
            batch_size=s["lidar"].shape[0]
        )
        tensordict = self.feature_extractor.feature_extractor(tensordict)
        q = self.qvalue(tensordict)["state_action_value"]
        return q


class SACModel(TensorDictModuleBase):
    """
    完整的 SAC 模型
    
    包含：
    1. Actor 网络
    2. 双 Critic 网络（Q1, Q2）
    3. Target Critic 网络（Q1_target, Q2_target）
    4. Temperature 参数（alpha）
    5. 优化器
    """
    
    def __init__(self, cfg, observation_spec, action_spec, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = observation_spec
        self.act_dim = action_spec
        self.device = device
        
        # 提取 action_dim
        if hasattr(self.act_dim, "shape"):
            shape = tuple(self.act_dim.shape)
            self.act_dim = int(shape[-1]) if len(shape) > 0 else int(shape[0])
        else:
            self.act_dim = int(self.act_dim)
        
        # 初始化网络
        print("🚀 Initializing SAC networks...")
        self.actor = ActorNetwork(self.obs_dim, self.act_dim, device).to(self.device)
        self.critic1 = CriticNetwork(self.obs_dim, self.act_dim, device).to(self.device)
        self.critic2 = CriticNetwork(self.obs_dim, self.act_dim, device).to(self.device)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        
        # Temperature 参数
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(5.0, device=device)))
        self.alpha = self.log_alpha.exp().detach()
        self.target_entropy = -float(self.act_dim)
        
        # 超参数
        self.gamma = getattr(cfg, 'gamma', 0.99)
        self.action_limit = getattr(cfg.actor, 'action_limit', 2.0)
        
        # 优化器
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=cfg.critic.learning_rate)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=cfg.critic.learning_rate)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_learning_rate)
        
        # 参数初始化
        self._init_weights()
        
        print(f"✅ SACModel initialized with action_dim={self.act_dim}")
    
    def _init_weights(self):
        """权重初始化（正交初始化）"""
        def init_(module):
            from torch.nn.parameter import UninitializedParameter
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
        print("✅ Weights initialized with orthogonal initialization")
    
    def get_action(self, state: Dict, deterministic: bool = True) -> torch.Tensor:
        """
        获取动作（推理接口）
        
        Args:
            state: 观测字典
            deterministic: 是否确定性输出（使用均值）
            
        Returns:
            action: 动作张量
        """
        if deterministic:
            with torch.no_grad():
                action, mu, log_std = self.actor(state)
        else:
            action, mu, log_std = self.actor(state)
        return action
    
    def __call__(self, td: TensorDict) -> TensorDict:
        """
        模型调用（环境交互接口）
        
        Args:
            td: TensorDict，包含观测
            
        Returns:
            td: 添加了动作的 TensorDict
        """
        td = td.to(self.device)
        action_n, mu, log_std = self.actor(td["agents", "observation"])
        actions_world = self.actions_to_world(action_n, td).squeeze(-1)
        td["agents", "action"] = actions_world
        td["agents", "action_normalized"] = action_n
        return td
    
    def train_step(self, replay_buffer, batch_size: int, tau: float = 0.005) -> Dict[str, float]:
        """
        SAC 训练步骤
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批大小
            tau: 软更新系数
            
        Returns:
            loss_info: 损失信息字典
        """
        train_tds = []
        
        for _ in range(self.cfg.num_minibatches):
            batch = replay_buffer.sample(batch_size).to(self.device)
            states = batch['agents', 'observation'].squeeze(-1).to(self.device)
            actions = batch['agents', 'action_normalized'].to(self.device)
            rewards = batch['next', 'agents', 'reward'].squeeze(1).to(self.device)
            next_states = batch['next', 'agents', 'observation'].squeeze(-1).to(self.device)
            dones = batch['next', 'terminated'].squeeze(-1).to(torch.bool).float().to(self.device)
            
            # ============ 更新 Critics ============
            with torch.no_grad():
                _, next_mu, next_log_std = self.actor(next_states)
                next_std = next_log_std.exp().clamp(min=1e-6)
                next_normal = torch.distributions.Normal(next_mu, next_std)
                next_x_t = next_normal.rsample()
                next_actions = torch.tanh(next_x_t)
                
                next_log_probs = next_normal.log_prob(next_x_t) - torch.log(1 - next_actions.pow(2) + 1e-6)
                next_log_probs = next_log_probs.sum(-1)
                
                next_q1 = self.critic1_target(next_states, next_actions).squeeze(-1)
                next_q2 = self.critic2_target(next_states, next_actions).squeeze(-1)
                next_q = torch.min(next_q1, next_q2)
                
                target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)
            
            q1 = self.critic1(states, actions).squeeze(-1)
            q2 = self.critic2(states, actions).squeeze(-1)
            
            critic1_loss = F.mse_loss(q1, target_q)
            critic2_loss = F.mse_loss(q2, target_q)
            
            self.critic1_optim.zero_grad()
            critic1_loss.backward()
            self.critic1_optim.step()
            
            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            self.critic2_optim.step()
            
            # ============ 更新 Actor ============
            _, mu, log_std = self.actor(states)
            std = log_std.exp().clamp(min=1e-6)
            normal = torch.distributions.Normal(mu, std)
            x_t = normal.rsample()
            actions_new = torch.tanh(x_t)
            
            log_probs = normal.log_prob(x_t) - torch.log(1 - actions_new.pow(2) + 1e-6)
            log_probs = log_probs.sum(-1)
            
            q1_new = self.critic1(states, actions_new).squeeze(-1)
            q2_new = self.critic2(states, actions_new).squeeze(-1)
            q_min = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha * log_probs - q_min).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            # ============ 更新 Temperature ============
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            
            # ============ 软更新 Target Networks ============
            self._soft_update(self.critic1_target, self.critic1, tau)
            self._soft_update(self.critic2_target, self.critic2, tau)
            
            # 记录训练信息
            train_td = TensorDict({
                "actor_loss": actor_loss.item(),
                "q1_loss": critic1_loss.item(),
                "q2_loss": critic2_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": self.alpha.item(),
                "actor_lp": log_probs.mean(),
                "q1": q1.mean(),
                "q_min": q_min.mean(),
                "q1_new": q1_new.mean(),
                "td_error": (q1_new - target_q).mean(),
                "td_error_target": (q1 - target_q).mean(),
            }, [])
            train_tds.append(train_td)
        
        loss_infos = torch.stack(train_tds).to_tensordict()
        loss_infos = loss_infos.apply(torch.mean, batch_size=[])
        return {k: v.mean().item() for k, v in loss_infos.items()}
    
    def actions_to_world(self, actions: torch.Tensor, tensordict: TensorDict) -> torch.Tensor:
        """将动作从局部坐标系转换到世界坐标系"""
        actions = actions * self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        return actions_world
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """软更新目标网络"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


class SACModelManager:
    """
    SAC 模型管理器
    
    提供统一的配置和状态管理接口，包括：
    1. 模型创建和初始化
    2. 检查点保存和加载
    3. wandb 集成
    4. 训练模式切换
    """
    
    def __init__(self, cfg, observation_spec, action_spec, device: torch.device):
        """
        初始化模型管理器
        
        Args:
            cfg: 配置对象
            observation_spec: 观测空间规格
            action_spec: 动作空间规格
            device: 设备（cpu/cuda）
        """
        self.cfg = cfg
        self.device = device
        
        # 创建模型
        self.model = SACModel(cfg, observation_spec, action_spec, device)
        
        print(f"✅ SACModelManager initialized on {device}")
    
    def save_checkpoint(self, path: Union[str, Path], step: int, **extra_info):
        """
        保存检查点
        
        Args:
            path: 保存路径
            step: 训练步数
            **extra_info: 额外信息（如 replay_buffer 状态）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'actor_optim_state_dict': self.model.actor_optim.state_dict(),
            'critic1_optim_state_dict': self.model.critic1_optim.state_dict(),
            'critic2_optim_state_dict': self.model.critic2_optim.state_dict(),
            'alpha_optim_state_dict': self.model.alpha_optim.state_dict(),
            'log_alpha': self.model.log_alpha.data,
            'cfg': dict(self.cfg),
            **extra_info
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved to {path}")
        
        # 上传到 wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"sac-model-step-{step}",
                type="model",
                metadata={"step": step}
            )
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)
            print(f"☁️  Checkpoint uploaded to wandb")
    
    def load_checkpoint(self, path: Union[str, Path], load_optimizers: bool = True) -> Dict:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            load_optimizers: 是否加载优化器状态
            
        Returns:
            checkpoint: 检查点字典
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizers:
            self.model.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
            self.model.critic1_optim.load_state_dict(checkpoint['critic1_optim_state_dict'])
            self.model.critic2_optim.load_state_dict(checkpoint['critic2_optim_state_dict'])
            self.model.alpha_optim.load_state_dict(checkpoint['alpha_optim_state_dict'])
        
        self.model.log_alpha.data = checkpoint['log_alpha']
        self.model.alpha = self.model.log_alpha.exp().detach()
        
        print(f"✅ Checkpoint loaded from {path} (step={checkpoint['step']})")
        return checkpoint
    
    def set_training_mode(self, mode: bool):
        """设置训练/评估模式"""
        self.model.train(mode)
    
    def get_action(self, state: Dict, deterministic: bool = True) -> torch.Tensor:
        """获取动作（推理接口）"""
        return self.model.get_action(state, deterministic)
    
    def __call__(self, td: TensorDict) -> TensorDict:
        """模型调用（环境交互接口）"""
        return self.model(td)
    
    def train_step(self, replay_buffer, batch_size: int, tau: float = 0.005) -> Dict[str, float]:
        """训练步骤"""
        return self.model.train_step(replay_buffer, batch_size, tau)
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'action_dim': self.model.act_dim,
            'gamma': self.model.gamma,
            'action_limit': self.model.action_limit,
            'alpha': self.model.alpha.item(),
            'target_entropy': self.model.target_entropy,
            'num_actor_params': sum(p.numel() for p in self.model.actor.parameters()),
            'num_critic_params': sum(p.numel() for p in self.model.critic1.parameters()),
        }


# 便捷函数
def create_sac_model(cfg, observation_spec, action_spec, device: torch.device) -> SACModelManager:
    """
    创建 SAC 模型管理器（便捷函数）
    
    Args:
        cfg: 配置对象
        observation_spec: 观测空间规格
        action_spec: 动作空间规格
        device: 设备
        
    Returns:
        model_manager: SAC 模型管理器
    """
    return SACModelManager(cfg, observation_spec, action_spec, device)
