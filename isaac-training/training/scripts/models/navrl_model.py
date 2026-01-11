'''
Author: zdytim zdytim@foxmail.com
Date: 2026-01-05 22:20:12
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-07 00:16:15
FilePath: /NavRL/isaac-training/training/scripts/models/navrl_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
NavRL Model Manager
===================
抽象的模型管理模块，将PPO-ViT模型的核心逻辑从训练脚本中分离出来。
提供统一的模型创建、加载、保存和配置管理接口。

主要功能：
1. SharedFeatureExtractor: ViT-based特征提取器，支持参数冻结/解冻
2. NavRLModel: 完整的PPO-ViT模型，包含Actor/Critic
3. ModelManager: 模型管理器，提供统一的配置和状态管理接口
4. 支持混合精度训练、分组优化器、参数管理等高级功能

作者: NavRL Team
日期: 2026年1月5日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torch.cuda.amp import autocast, GradScaler
import os
import tempfile
import json
from pathlib import Path
import torch.nn.utils.spectral_norm as spectral_norm
from typing import Dict, Optional, Tuple, List, Union

# wandb integration for model management
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Model uploading will be disabled.")

# 导入项目依赖
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world
from VIT import VIT


class SharedFeatureExtractor(nn.Module):
    """
    共享特征提取器
    
    功能：
    1. ViT backbone用于视觉特征提取
    2. 动态障碍物编码器
    3. 状态编码器
    4. 特征融合网络
    5. 支持选择性参数冻结/解冻
    """
    
    def __init__(self, device: torch.device, pretrained_checkpoint_path: Optional[str] = None, input_size: tuple = (224, 224)):
        super().__init__()
        self.device = device
        self.input_size = input_size
        
        # ViT Backbone with dynamic sizing
        print(f"🧠 Initializing ViT backbone for {input_size}...")
        self.vit = VIT(input_size=input_size).to(device)
        
        # 参数管理
        self._load_vit_weights(pretrained_checkpoint_path)
        self._setup_parameter_training()
        
        # 其他组件初始化
        self._init_other_modules()
        
        print(f"✅ SharedFeatureExtractor initialized on {device} for {input_size}")

    def forward(self, camera: torch.Tensor, dynamic_obstacle: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # , hide: torch.Tensor
        """
        前向传播
        
        Args:
            camera: 相机输入 [Batch, 1, H, W]
            dynamic_obstacle: 动态障碍物 [Batch, C, W, H]  
            state: 状态信息 [Batch, 8]
            
        Returns:
            latent: 融合特征 [Batch, 128]
        """
        # 1. 相机数据预处理
        camera = torch.nan_to_num(camera, nan=10.0, posinf=10.0, neginf=0.0)
        camera = camera.clamp(0.0, 10.0)
        
        assert camera.dim() == 4, f"Camera input should be 4D [B,C,H,W], got {camera.shape}"
        assert camera.shape[1] == 1, f"Camera input should be grayscale (1 channel), got {camera.shape[1]} channels"
        
        # 归一化处理
        x = (camera / 10 if camera.max() > 1.1 else camera)
        
        # 2. ViT特征提取（冻结状态）
        # with torch.no_grad():
        v_feat = self.vit(x)  # [Batch, 512]
        
        # 3. 动态障碍物特征提取
        d_feat = self.dyn_ext(dynamic_obstacle)  # [Batch, 64]
        
        # 4. 状态特征提取
        s_feat = self.state_ext(state)  # [Batch, 64]
        
        # 5. 特征融合
        combined = torch.cat([v_feat, d_feat, state], dim=-1)  # [Batch, 584]
        # if len(hide) == 1:
        #     latent, h = self.lstm(combined, hide)
        # else:
        #     latent, h = self.lstm(combined)
        # latent = self.nn_fc2(latent)
        latent = self.fusion_mlp(combined)  # [Batch, 128]

        return latent

    def _load_vit_weights(self, checkpoint_path: Optional[str]) -> None:
        """加载ViT预训练权重"""
        if checkpoint_path is None:
            print("⚠️  No ViT checkpoint provided, using random initialization")
            return
            
        try:
            print(f"🔄 Loading ViT weights from: {checkpoint_path}")
            full_model_checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 过滤ViT相关权重
            vit_state_dict = {}
            ignored_keys = []
            for k, v in full_model_checkpoint.items():
                if k.startswith('encoder_blocks') or k.startswith('decoder') or \
                   k.startswith('up_sample') or k.startswith('pxShuffle') or k.startswith('down_sample'):
                    vit_state_dict[k] = v
                else:
                    ignored_keys.append(k)
            
            # 加载权重
            missing_keys, unexpected_keys = self.vit.load_state_dict(vit_state_dict, strict=False)
            
            print(f"✅ ViT weights loaded successfully!")
            print(f"   - Loaded keys: {len(vit_state_dict)}")
            print(f"   - Ignored keys: {len(ignored_keys)}")
            if missing_keys:
                print(f"   - Missing keys: {len(missing_keys)} (will use random init)")
            if unexpected_keys:
                print(f"   - Unexpected keys: {len(unexpected_keys)}")
                
        except Exception as e:
            print(f"❌ Failed to load ViT weights: {e}")
            print("   - Using random initialization")
    
    def _setup_parameter_training(self) -> None:
        """设置参数训练状态：冻结encoder，解冻decoder"""
        # 1. 全部冻结
        self._freeze_all_vit_parameters()
        
        # 2. 选择性解冻
        self._unfreeze_decoder_modules()
    
    def _freeze_all_vit_parameters(self) -> None:
        """冻结所有ViT参数"""
        frozen_params = 0
        for param in self.vit.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"❄️  Frozen all ViT parameters: {frozen_params:,}")
    
    def _unfreeze_decoder_modules(self) -> None:
        """解冻decoder模块进行fine-tuning"""
        decoder_modules = ['encoder_blocks', 'decoder', 'up_sample', 'pxShuffle']
        unfrozen_params = 0
        
        print("🔓 Unfreezing ViT decoder modules for fine-tuning:")
        for module_name in decoder_modules:
            if hasattr(self.vit, module_name):
                module = getattr(self.vit, module_name)
                for param in module.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                print(f"   - {module_name}: ✅ unfrozen")
            else:
                print(f"   - {module_name}: ⚠️ not found in model")
        
        print(f"   - Total decoder parameters: {unfrozen_params:,}")
    
    def _init_other_modules(self) -> None:
        """初始化其他网络模块"""
        # 动态障碍物提取器
        self.dyn_ext = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(50, 128), nn.LeakyReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.LayerNorm(64),
        ).to(self.device)

        # 状态特征提取器
        self.state_ext = nn.Sequential(
            nn.Linear(8, 64), nn.LeakyReLU(), nn.LayerNorm(64),
            nn.Linear(64, 64), nn.LeakyReLU(), nn.LayerNorm(64),
        ).to(self.device)

        # 融合网络
        # self.lstm = (nn.LSTM(input_size=584, hidden_size=128,num_layers=3, dropout=0.1)).to(self.device)
        # self.nn_fc2 = spectral_norm(nn.Linear(128, 3)).to(self.device)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(584, 512), nn.LeakyReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.LeakyReLU(), nn.LayerNorm(128)
        ).to(self.device)
    
    # === 高级参数管理接口 ===
    
    def freeze_vit_encoder(self) -> None:
        """冻结ViT encoder（保持decoder解冻状态）"""
        encoder_modules = ['encoder_blocks']
        frozen_params = 0
        
        for module_name in encoder_modules:
            if hasattr(self.vit, module_name):
                module = getattr(self.vit, module_name)
                for param in module.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
        
        print(f"❄️  Frozen encoder: {frozen_params:,} parameters")
    
    def unfreeze_all_vit(self) -> None:
        """解冻所有ViT参数（用于完全fine-tuning）"""
        unfrozen_params = 0
        for param in self.vit.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        
        print(f"🔓 Unfrozen all ViT: {unfrozen_params:,} parameters")
    
    def get_parameter_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """
        获取参数组用于优化器配置
        
        Returns:
            参数组字典，包含ViT和其他模块的参数
        """
        vit_params = [p for p in self.vit.parameters() if p.requires_grad]
        other_params = []
        
        for module in [self.dyn_ext, self.state_ext, self.fusion_mlp]:
            other_params.extend([p for p in module.parameters() if p.requires_grad])
        
        return {
            'vit_decoder': vit_params,
            'other_modules': other_params
        }
    
    def get_parameter_stats(self) -> Dict[str, int]:
        """获取参数统计信息"""
        param_groups = self.get_parameter_groups()
        return {
            'vit_decoder_count': sum(p.numel() for p in param_groups['vit_decoder']),
            'other_modules_count': sum(p.numel() for p in param_groups['other_modules']),
            'total_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class NavRLModel(TensorDictModuleBase):
    """
    NavRL完整模型
    
    包含：
    1. SharedFeatureExtractor: 特征提取
    2. Actor Head: 策略网络
    3. Critic Head: 价值网络
    4. 优化器和训练工具
    """
    
    def __init__(self, cfg, observation_spec, action_spec, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # 处理action_spec
        if hasattr(action_spec, "shape"):
            shape = tuple(action_spec.shape)
            self.action_dim = int(shape[-1]) if len(shape) > 0 else int(shape[0])
        else:
            self.action_dim = int(action_spec)
        
        # 初始化网络组件
        self._init_networks(observation_spec)
        
        # 初始化训练工具
        self._init_training_tools()
        
        print(f"✅ NavRLModel initialized with {self.action_dim}D actions")
    
    def _init_networks(self, observation_spec) -> None:
        """初始化网络组件"""
        # 1. 从配置获取输入尺寸
        input_size = getattr(self.cfg.feature_extractor, 'input_size', (224, 224))
        if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
            input_size = tuple(input_size)
        else:
            input_size = (224, 224)  # 默认值
            
        # 2. 共享特征提取器
        pretrained_path = getattr(self.cfg.feature_extractor, 'pretrained_checkpoint_path', None)
        self.shared_features = SharedFeatureExtractor(
            self.device, 
            pretrained_path,
            input_size=input_size  # 传递配置的输入尺寸
        )
        
        # ... 其余初始化代码保持不变 ...
        
        print(f"✅ NavRLModel initialized with input size: {input_size}")

        # 2. Actor Head
        self.actor_head = ProbabilisticActor(
            TensorDictModule(
                BetaActor(self.action_dim), 
                in_keys=["_latent"], 
                out_keys=["alpha", "beta"]
            ),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # 3. Critic Head
        self.critic_head = nn.Linear(128, 1).to(self.device)
        
        # 4. 初始化网络权重
        self._init_dummy_forward(observation_spec)
        self._init_weights()
    
    def _init_dummy_forward(self, observation_spec) -> None:
        """执行dummy forward以初始化LazyLinear"""
        dummy_tensordict = observation_spec.zero().unsqueeze(0).to(self.device).reshape(-1)
        
        with torch.no_grad():
            latent = self.shared_features(
                dummy_tensordict["agents", "observation", "camera"],
                dummy_tensordict["agents", "observation", "dynamic_obstacle"],
                dummy_tensordict["agents", "observation", "state"]
            )
            dummy_tensordict.set("_latent", latent)
            self.actor_head(dummy_tensordict)
    
    def _init_weights(self) -> None:
        """初始化Actor和Critic权重"""
        def init_(m):
            if isinstance(m, nn.Linear):
                weight = getattr(m, "weight", None)
                if isinstance(weight, torch.nn.parameter.UninitializedParameter):
                    return
                nn.init.orthogonal_(weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
        
        print("🔄 Initializing Actor and Critic weights...")
        self.actor_head.apply(init_)
        self.critic_head.apply(init_)
        print("   - Network weights initialized ✅")
    
    def _init_training_tools(self) -> None:
        """初始化训练工具"""
        # 1. 优化器（分组学习率）
        self.optimizer = self._create_grouped_optimizer()
        
        # 2. 训练工具
        self.gae = GAE(0.99, 0.95)
        self.value_norm = ValueNorm(1).to(self.device)
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        
        # 3. 混合精度训练
        self.use_amp = getattr(self.cfg, 'use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        if self.use_amp:
            print("✅ Mixed Precision (AMP) enabled - saving 30-50% GPU memory")
    
    def _create_grouped_optimizer(self) -> torch.optim.Optimizer:
        """创建分组优化器"""
        param_groups = []
        
        # 1. ViT decoder参数（低学习率）
        feature_groups = self.shared_features.get_parameter_groups()
        vit_params = feature_groups['vit_decoder']
        
        if vit_params:
            decoder_lr = self.cfg.actor.learning_rate * 0.1  # 10倍降低
            param_groups.append({
                'params': vit_params, 
                'lr': decoder_lr,
                'name': 'vit_decoder'
            })
            print(f"📚 ViT decoder: {sum(p.numel() for p in vit_params):,} params @ lr={decoder_lr}")
        
        # 2. 其他模块参数（正常学习率）
        other_params = feature_groups['other_modules']
        other_params.extend([p for p in self.actor_head.parameters() if p.requires_grad])
        other_params.extend([p for p in self.critic_head.parameters() if p.requires_grad])
        
        param_groups.append({
            'params': other_params, 
            'lr': self.cfg.actor.learning_rate,
            'name': 'task_specific'
        })
        
        print(f"🎯 Task-specific: {sum(p.numel() for p in other_params):,} params @ lr={self.cfg.actor.learning_rate}")
        print(f"📊 Total trainable: {sum(p.numel() for p in vit_params + other_params):,} parameters")
        
        return torch.optim.Adam(param_groups)
    
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        """推理模式：用于环境交互"""
        # 1. 特征提取
        latent = self.shared_features(
            tensordict["agents", "observation", "camera"],
            tensordict["agents", "observation", "dynamic_obstacle"],
            tensordict["agents", "observation", "state"]
        )
        tensordict.set("_latent", latent)
        
        # 2. 策略采样
        self.actor_head(tensordict)
        
        # 3. 价值估计
        value = self.critic_head(latent)
        tensordict.set("state_value", value)

        # 4. 坐标转换 (Local -> World)
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        
        return tensordict
    
    def get_model_info(self) -> Dict[str, Union[int, str]]:
        """获取模型信息"""
        feature_stats = self.shared_features.get_parameter_stats()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'vit_decoder_params': feature_stats['vit_decoder_count'],
            'other_modules_params': feature_stats['other_modules_count'],
            'action_dim': self.action_dim,
            'device': str(self.device),
            'use_amp': self.use_amp
        }


class ModelManager:
    """
    模型管理器
    
    提供统一的模型创建、加载、保存和配置管理接口
    """
    
    def __init__(self, cfg, observation_spec, action_spec, device: torch.device):
        self.cfg = cfg
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.device = device
        
        # 创建模型
        self.model = NavRLModel(cfg, observation_spec, action_spec, device)
        
        print("🎉 ModelManager initialized successfully")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> bool:
        """
        加载完整检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            load_optimizer: 是否加载优化器状态
            
        Returns:
            加载是否成功
        """
        try:
            print(f"🔄 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict'] 
            else:
                state_dict = checkpoint
            
            # 统计加载结果
            loaded_stats = self._load_model_state(state_dict)
            
            # 加载训练状态
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                try:
                    self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("   📈 Optimizer state: ✅ loaded")
                except Exception as e:
                    print(f"   📈 Optimizer state: ❌ failed ({e})")
            
            if 'value_norm_state' in checkpoint:
                try:
                    self.model.value_norm.load_state_dict(checkpoint['value_norm_state'])
                    print("   📊 Value normalization: ✅ loaded")
                except Exception as e:
                    print(f"   📊 Value normalization: ❌ failed ({e})")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def _load_model_state(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """加载模型状态并统计结果"""
        loaded_stats = {
            'shared_features_vit': 0,
            'shared_features_other': 0,
            'actor_head': 0, 
            'critic_head': 0,
            'skipped': 0
        }
        
        current_state = self.model.state_dict()
        matched_params = {}
        
        for name, param in state_dict.items():
            if name in current_state:
                if current_state[name].shape == param.shape:
                    matched_params[name] = param
                    
                    # 分类统计
                    if name.startswith('shared_features.vit.'):
                        loaded_stats['shared_features_vit'] += 1
                    elif name.startswith('shared_features.'):
                        loaded_stats['shared_features_other'] += 1
                    elif name.startswith('actor_head.'):
                        loaded_stats['actor_head'] += 1
                    elif name.startswith('critic_head.'):
                        loaded_stats['critic_head'] += 1
                else:
                    loaded_stats['skipped'] += 1
            else:
                loaded_stats['skipped'] += 1
        
        # 加载匹配的权重
        self.model.load_state_dict(matched_params, strict=False)
        
        # 打印统计结果
        total_loaded = sum(v for k, v in loaded_stats.items() if k != 'skipped')
        print(f"✅ Loaded {total_loaded} parameters:")
        
        if loaded_stats['shared_features_vit'] > 0:
            print(f"   🧠 ViT backbone: {loaded_stats['shared_features_vit']} parameters")
        if loaded_stats['shared_features_other'] > 0:
            print(f"   ⚙️  Other features: {loaded_stats['shared_features_other']} parameters")
        if loaded_stats['actor_head'] > 0:
            print(f"   🎯 Actor head: {loaded_stats['actor_head']} parameters")
        if loaded_stats['critic_head'] > 0:
            print(f"   🎯 Critic head: {loaded_stats['critic_head']} parameters")
        if loaded_stats['skipped'] > 0:
            print(f"   ⚠️  Skipped: {loaded_stats['skipped']} parameters")
        
        return loaded_stats
    
    def save_checkpoint(self, filepath: str, epoch: int = 0, step: int = 0, 
                       additional_info: Optional[Dict] = None, 
                       upload_to_wandb: bool = False,
                       wandb_alias: Optional[str] = None) -> None:
        """
        保存检查点并可选上传到wandb
        
        Args:
            filepath: 保存路径
            epoch: 训练轮次
            step: 训练步数
            additional_info: 额外信息
            upload_to_wandb: 是否上传到wandb
            wandb_alias: wandb模型版本别名
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'value_norm_state': self.model.value_norm.state_dict(),
            'model_config': self.cfg,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        print(f"💾 Checkpoint saved: {filepath}")
        
        # 可选：上传到wandb
        if upload_to_wandb and WANDB_AVAILABLE:
            self._upload_to_wandb(filepath, step, wandb_alias, additional_info)
    
    def _upload_to_wandb(self, filepath: str, step: int, alias: Optional[str] = None, 
                        metadata: Optional[Dict] = None) -> None:
        """上传模型到wandb"""
        try:
            if not wandb.run:
                print("⚠️  No active wandb run. Cannot upload model.")
                return
            
            # 获取模型信息
            model_info = self.model.get_model_info()
            
            # 创建artifact
            model_name = f"navrl-model-step-{step}"
            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                metadata={
                    'step': step,
                    'architecture': 'PPO-ViT',
                    'total_parameters': model_info['total_parameters'],
                    'trainable_parameters': model_info['trainable_parameters'],
                    'action_dim': model_info['action_dim'],
                    'device': model_info['device'],
                    'use_amp': model_info['use_amp'],
                    **(metadata or {})
                }
            )
            
            # 添加模型文件
            artifact.add_file(filepath)
            
            # 创建模型卡片
            model_card_path = self._create_model_card(filepath, model_info, step, metadata)
            artifact.add_file(model_card_path, name="model_card.md")
            
            # 记录artifact
            wandb.log_artifact(artifact, aliases=[alias] if alias else None)
            
            print(f"📤 Model uploaded to wandb: {model_name}")
            if alias:
                print(f"   🏷️  Alias: {alias}")
            
            # 清理临时文件
            if os.path.exists(model_card_path) and "tmp" in model_card_path:
                os.remove(model_card_path)
                
        except Exception as e:
            print(f"❌ Failed to upload model to wandb: {e}")
    
    def _create_model_card(self, filepath: str, model_info: Dict, step: int, 
                          metadata: Optional[Dict] = None) -> str:
        """创建模型卡片"""
        # 创建临时模型卡片文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(f"""# NavRL Model Card - Step {step}

## Model Overview
- **Architecture**: PPO-ViT with Shared Feature Extractor  
- **Training Step**: {step:,}
- **File Size**: {os.path.getsize(filepath) / (1024**2):.2f} MB

## Architecture Details
- **Total Parameters**: {model_info['total_parameters']:,}
- **Trainable Parameters**: {model_info['trainable_parameters']:,}  
- **Frozen Parameters**: {model_info['frozen_parameters']:,}
- **Action Dimension**: {model_info['action_dim']}
- **Mixed Precision**: {'Enabled' if model_info['use_amp'] else 'Disabled'}

## Component Breakdown
- **ViT Decoder**: {model_info['vit_decoder_params']:,} parameters
- **Other Modules**: {model_info['other_modules_params']:,} parameters

## Training Configuration
- **Device**: {model_info['device']}
- **Optimizer**: Adam with grouped learning rates
- **Loss Function**: PPO with value clipping

""")

            # 添加额外的元数据信息
            if metadata:
                f.write("## Training Metrics\n")
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n")
                    else:
                        f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                f.write("\n")
            
            f.write("""## Usage
```python
# 加载模型
from models import load_pretrained_model
model_manager = load_pretrained_model(checkpoint_path, cfg, obs_spec, act_spec, device)
model = model_manager.get_model()

# 推理
output = model(input_tensordict)
```

## Model Components
1. **SharedFeatureExtractor**: ViT-based visual feature extraction
2. **Actor Head**: Policy network with Beta distribution
3. **Critic Head**: Value function estimation
4. **Optimization**: Grouped learning rates for ViT fine-tuning

Generated by NavRL ModelManager
""")
            return f.name
    
    def upload_model_to_registry(self, model_name: str, description: str = "",
                                tags: Optional[List[str]] = None,
                                step: int = 0) -> None:
        """
        上传模型到wandb模型注册表
        
        Args:
            model_name: 模型名称
            description: 模型描述  
            tags: 标签列表
            step: 训练步数
        """
        if not WANDB_AVAILABLE or not wandb.run:
            print("⚠️  wandb not available or no active run.")
            return
        
        try:
            # 创建临时保存路径
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                temp_path = f.name
            
            # 保存模型
            self.save_checkpoint(temp_path, step=step)
            
            # 获取模型信息
            model_info = self.model.get_model_info()
            
            # 创建模型artifact用于注册表
            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                description=description,
                metadata={
                    'framework': 'NavRL',
                    'architecture': 'PPO-ViT',
                    'step': step,
                    **model_info
                }
            )
            
            # 添加文件
            artifact.add_file(temp_path, name="model.pt")
            
            # 创建和添加模型卡片
            model_card_path = self._create_model_card(temp_path, model_info, step)
            artifact.add_file(model_card_path, name="README.md")
            
            # 记录到注册表
            wandb.run.log_artifact(artifact, aliases=tags or ["latest"])
            
            print(f"🎯 Model '{model_name}' uploaded to wandb registry")
            
            # 清理临时文件
            os.remove(temp_path)
            if os.path.exists(model_card_path):
                os.remove(model_card_path)
                
        except Exception as e:
            print(f"❌ Failed to upload to registry: {e}")
    
    def load_from_wandb(self, artifact_path: str, load_optimizer: bool = True) -> bool:
        """
        从wandb artifact加载模型
        
        Args:
            artifact_path: wandb artifact路径 (e.g., "username/project/model-name:version")
            load_optimizer: 是否加载优化器状态
            
        Returns:
            加载是否成功
        """
        if not WANDB_AVAILABLE:
            print("❌ wandb not available")
            return False
        
        try:
            print(f"🔄 Loading model from wandb: {artifact_path}")
            
            # 下载artifact
            artifact = wandb.use_artifact(artifact_path)
            artifact_dir = artifact.download()
            
            # 寻找模型文件
            model_files = list(Path(artifact_dir).glob("*.pt"))
            if not model_files:
                print("❌ No .pt model file found in artifact")
                return False
            
            model_path = str(model_files[0])
            print(f"   📂 Model file: {model_path}")
            
            # 加载模型
            success = self.load_checkpoint(model_path, load_optimizer)
            
            if success:
                print(f"✅ Model loaded from wandb successfully")
                # 打印artifact元数据
                if hasattr(artifact, 'metadata') and artifact.metadata:
                    print("   📋 Artifact metadata:")
                    for key, value in artifact.metadata.items():
                        print(f"      {key}: {value}")
            
            return success
            
        except Exception as e:
            print(f"❌ Failed to load from wandb: {e}")
            return False
    
    def print_model_summary(self) -> None:
        """打印模型摘要"""
        info = self.model.get_model_info()
        
        print("\n" + "="*60)
        print("📊 NavRL Model Summary")
        print("="*60)
        print(f"🏗️  Architecture: PPO-ViT with Shared Feature Extractor")
        print(f"🎯 Action Dimension: {info['action_dim']}")
        print(f"💾 Device: {info['device']}")
        print(f"⚡ Mixed Precision: {'Enabled' if info['use_amp'] else 'Disabled'}")
        print(f"📈 Total Parameters: {info['total_parameters']:,}")
        print(f"🔄 Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"❄️  Frozen Parameters: {info['frozen_parameters']:,}")
        print(f"   - ViT Decoder: {info['vit_decoder_params']:,}")
        print(f"   - Other Modules: {info['other_modules_params']:,}")
        print("="*60 + "\n")
    
    def get_model(self) -> NavRLModel:
        """获取模型实例"""
        return self.model
    
    def set_training_mode(self, mode: bool = True) -> None:
        """设置训练/评估模式"""
        self.model.train(mode)
        if mode:
            print("🏃 Model set to TRAINING mode")
        else:
            print("🔍 Model set to EVALUATION mode")
    
    def freeze_vit_encoder(self) -> None:
        """冻结ViT encoder"""
        self.model.shared_features.freeze_vit_encoder()
    
    def unfreeze_all_vit(self) -> None:
        """解冻所有ViT参数"""
        self.model.shared_features.unfreeze_all_vit()
        # 重新创建优化器以包含新的可训练参数
        self.model.optimizer = self.model._create_grouped_optimizer()
        print("🔄 Optimizer updated with unfrozen parameters")


# === 工厂函数 ===

def create_navrl_model(cfg, observation_spec, action_spec, device: torch.device) -> ModelManager:
    """
    工厂函数：创建NavRL模型管理器
    
    Args:
        cfg: 配置对象
        observation_spec: 观察空间规格
        action_spec: 动作空间规格  
        device: 计算设备
        
    Returns:
        ModelManager实例
    """
    return ModelManager(cfg, observation_spec, action_spec, device)


def load_pretrained_model(checkpoint_path: str, cfg, observation_spec, action_spec, 
                         device: torch.device, load_optimizer: bool = True) -> ModelManager:
    """
    工厂函数：加载预训练模型
    
    Args:
        checkpoint_path: 检查点路径
        cfg: 配置对象
        observation_spec: 观察空间规格
        action_spec: 动作空间规格
        device: 计算设备
        load_optimizer: 是否加载优化器状态
        
    Returns:
        ModelManager实例
    """
    manager = create_navrl_model(cfg, observation_spec, action_spec, device)
    success = manager.load_checkpoint(checkpoint_path, load_optimizer)
    
    if not success:
        print("⚠️  继续使用随机初始化的模型")
    
    return manager


if __name__ == "__main__":
    # 示例使用
    print("NavRL Model Manager - 独立测试")
    
    # 这里可以添加独立的测试代码
    pass