'''
Author: zdytim zdytim@foxmail.com
Date: 2025-12-29 21:00:25
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-07 13:14:07
FilePath: /u20/NavRL/isaac-training/training/scripts/ppo_vit_v1 copy.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: zdytim zdytim@foxmail.com
Date: 2025-12-23
Description: PPO with ViT Backbone (SAC-style dimension handling & Shared Features)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world
from VIT import VIT
from torch.cuda.amp import autocast, GradScaler  # ✅ 混合精度训练


# ==========================================
# 2. 共享特征提取器 (纯净模型，不含维度处理)
# ==========================================
class SharedFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # --- A. ViT Backbone (DINO Pretrained & Frozen) ---
        print("Loading ViT backbone")
        self.vit = VIT().to(device)
        
        # 加载完整模型权重
        full_model_checkpoint = torch.load("/home/u20/NavRL/isaac-training/training/scripts/ViTLSTM_model.pth")
        
        # 只加载 ViT 相关的权重（过滤掉 LSTM 和 fc2）
        vit_state_dict = {}
        ignored_keys = []
        for k, v in full_model_checkpoint.items():
            # 保留 encoder_blocks、decoder 以及辅助层的权重
            if k.startswith('encoder_blocks') or k.startswith('decoder') or \
               k.startswith('up_sample') or k.startswith('pxShuffle') or k.startswith('down_sample'):
                vit_state_dict[k] = v
            else:
                ignored_keys.append(k)
        
        # 加载权重并记录结果
        missing_keys, unexpected_keys = self.vit.load_state_dict(vit_state_dict, strict=False)
        
        print(f"✅ ViT weights loaded successfully!")
        print(f"   - Loaded keys: {len(vit_state_dict)}")
        print(f"   - Ignored keys (LSTM/FC2): {ignored_keys}")
        if missing_keys:
            print(f"   ⚠️  Missing keys (will use random init): {missing_keys}")
        if unexpected_keys:
            print(f"   ⚠️  Unexpected keys: {unexpected_keys}")
        
        # 强制冻结参数
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval() # 始终保持 Eval 模式 (关闭 Dropout/BatchNorm更新)
        

        # --- B. 动态障碍物提取器 ---
        self.dyn_ext = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(50, 128), nn.LeakyReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.LayerNorm(64),
        ).to(device)

        # --- C. State 特征提取器（新增）---
        # 将 8 维原始 state 编码为 64 维特征，与 Dyn Obs 维度对齐
        self.state_ext = nn.Sequential(
            nn.Linear(8, 64), nn.LeakyReLU(), nn.LayerNorm(64),
            nn.Linear(64, 64), nn.LeakyReLU(), nn.LayerNorm(64),
        ).to(device)

        # --- D. 融合 MLP (漏斗结构) ---
        # Input: 512(ViT) + 64(Dyn) + 64(State) = 640
        self.fusion_mlp = nn.Sequential(
            nn.Linear(640, 512), nn.LeakyReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.LeakyReLU(), nn.LayerNorm(128)
        ).to(device)

    def forward(self, camera, dynamic_obstacle, state):
        """
        Input shapes are assumed to be flattened: [Batch, ...] 
        No dimension checks inside the model.
        """
        camera = torch.nan_to_num(camera, nan=10.0, posinf=10.0, neginf=0.0)
        camera = camera.clamp(0.0, 10.0)
        
        # 验证输入是单通道灰度图 [Batch, 1, H, W]
        assert camera.dim() == 4, f"Camera input should be 4D [B,C,H,W], got {camera.shape}"
        assert camera.shape[1] == 1, f"Camera input should be grayscale (1 channel), got {camera.shape[1]} channels"
        
        # 1. Image Processing
        x = (camera / 10 if camera.max() > 1.1 else camera)
        
        # [FIX] 转换 RGB 为单通道 (Gray)，因为 VIT 模型定义为 1 通道输入
        # x = (x - self.mean) / self.std
        
        with torch.no_grad():
            # ViT forward
            v_feat = self.vit(x) # [Batch, 512]
        
        # 2. Dynamic Obstacle Processing
        d_feat = self.dyn_ext(dynamic_obstacle) # [Batch, 64]
        
        # 3. State Processing (NEW: encode state for dimension balance)
        s_feat = self.state_ext(state) # [Batch, 8] -> [Batch, 64]
        
        # 4. Concatenation & Fusion (640 = 512+64+64)
        combined = torch.cat([v_feat, d_feat, s_feat], dim=-1) # [Batch, 640]
        latent = self.fusion_mlp(combined) # [Batch, 128]
        
        return latent

# ==========================================
# 3. PPO 主类
# ==========================================
class PPOVIT(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # 处理 action_spec 兼容性
        if hasattr(action_spec, "shape"):
            shape = tuple(action_spec.shape)
            self.action_dim = int(shape[-1]) if len(shape) > 0 else int(shape[0])
        else:
            self.action_dim = int(action_spec)

        # --- 初始化网络组件 ---
        self.shared_features = SharedFeatureExtractor(device)
        
        # Actor Head (Input: _latent -> Output: alpha, beta)
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
        ).to(device)

        # Critic Head (Input: _latent -> Output: state_value)
        self.critic_head = nn.Linear(128, 1).to(device)


        # 1. 构造一个 Dummy Input (假数据)
        # 从 observation_spec 中生成全0数据，并添加 Batch 维度 [1, ...]
        dummy_tensordict = observation_spec.zero().unsqueeze(0).to(device).reshape(-1)
    
        with torch.no_grad():
            latent = self.shared_features(
                dummy_tensordict["agents", "observation", "camera"],
                dummy_tensordict["agents", "observation", "dynamic_obstacle"],
                dummy_tensordict["agents", "observation", "state"]
            )
            dummy_tensordict.set("_latent", latent)
            # 运行一次 actor_head，触发 LazyLinear 初始化
            self.actor_head(dummy_tensordict)

        # ----------------------------------------------------------------
        # [原有逻辑] 现在的权重已经实例化了，可以安全初始化了
        # ----------------------------------------------------------------
        # --- 优化器与工具 ---
        # 统一优化器：只包含可训练的参数
        # 注意：ViT 参数已被冻结（requires_grad=False），不会包含在优化器中
        all_params = []
        for module in [self.shared_features, self.actor_head, self.critic_head]:
            all_params.extend([p for p in module.parameters() if p.requires_grad])
        
        print(f"📊 Trainable parameters: {sum(p.numel() for p in all_params):,}")
        print(f"   - Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        
        self.optimizer = torch.optim.Adam(all_params, lr=cfg.actor.learning_rate)
        self.gae = GAE(0.99, 0.95)
        self.value_norm = ValueNorm(1).to(device)
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        
        # 混合精度训练（FP16）- 可节省 30-50% 显存
        self.use_amp = getattr(cfg, 'use_amp', True)  # 默认启用
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("✅ Mixed Precision (AMP) enabled - saving 30-50% GPU memory")

        # 现在调用初始化不会报错了
        self._init_weights()

    def _init_weights(self):
        def init_(m):
            if isinstance(m, nn.Linear):
                weight = getattr(m, "weight", None)
                if isinstance(weight, torch.nn.parameter.UninitializedParameter):
                    return
                nn.init.orthogonal_(weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
        self.actor_head.apply(init_)
        self.critic_head.apply(init_)

    def __call__(self, tensordict):
        """推理模式：通常用于环境交互"""
        # 1. 提取特征
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
        
        # ✅ 清理中间变量（可选，提升 collector 性能）
        # 注释掉这行如果需要调试中间变量
        # for key in ["_latent", "alpha", "beta"]:
        #     if key in tensordict.keys():
        #         tensordict.exclude(key, inplace=True)
        
        return tensordict

    def train(self, tensordict):
        """
        训练函数：
        在外部处理所有维度问题，模拟 SAC 的 buffer.sample() 效果。
        """
        # 1. 获取维度信息 [Batch, Time]
        # 即使 T=1，Tensordict 依然保留这个结构
        B, T = tensordict.shape 
        
        # step 1: 展平 Batch 维度供网络推理
        td_flat = tensordict.reshape(-1)
        next_td_flat = tensordict["next"].reshape(-1)

        with torch.no_grad():
            # -----------------------------------------------------------------
            # [方案1] 使用采样时已保存的values（标准PPO做法）
            # -----------------------------------------------------------------
            # 1. 从tensordict读取采样时保存的values
            values_flat = tensordict["state_value"].reshape(-1, 1)  # [B*T, 1]
            
            # 2. 只需计算next_values
            next_latent = self.shared_features(
                next_td_flat["agents", "observation", "camera"],
                next_td_flat["agents", "observation", "dynamic_obstacle"],
                next_td_flat["agents", "observation", "state"]
            )
            next_values_flat = self.critic_head(next_latent)  # [B*T, 1]
            
            # 3. 反归一化
            values_flat = self.value_norm.denormalize(values_flat)
            next_values_flat = self.value_norm.denormalize(next_values_flat)

            # 4. 还原为 [B, T] 供 GAE 使用
            values = values_flat.squeeze(-1).view(B, T)
            next_values = next_values_flat.squeeze(-1).view(B, T)
            # -----------------------------------------------------------------

        # step 3: 准备 GAE 所需数据
        # 原始数据是 [B, T, 1]，squeeze 掉最后一维变成 [B, T]
        rewards = tensordict["next", "agents", "reward"].squeeze(-1) 
        dones = tensordict["next", "terminated"].float().squeeze(-1)

        # step 4: 计算 GAE
        # 输入全部为 [B, T]，values 和 next_values 是反归一化后的真实值
        # GAE 返回的 adv 和 ret 也都是真实尺度
        adv, ret = self.gae(rewards, dones, values, next_values)
        
        # step 5: ValueNorm 更新和归一化
        # 5.1 用真实 Return 更新统计信息
        ret_flat = ret.reshape(-1, 1)  # [B*T, 1]
        self.value_norm.update(ret_flat)
        
        # 5.2 归一化 Return 用于 Critic Loss Target
        ret_normalized_flat = self.value_norm.normalize(ret_flat)  # [B*T, 1]
        
        # 5.3 还原形状
        ret_normalized = ret_normalized_flat.view(B, T, 1)
        
        # step 6: 将计算结果存回 tensordict
        # 标准化 Advantage（全局归一化）
        adv_normalized = (adv - adv.mean()) / adv.std().clip(1e-7)
        
        # 归一化采样时的 value（用于 critic loss clipping）
        values_normalized = self.value_norm.normalize(values.unsqueeze(-1).reshape(-1, 1))
        values_normalized = values_normalized.view(B, T, 1)
        
        tensordict.set("adv", adv_normalized.unsqueeze(-1))  # [B, T, 1] - 归一化后的
        tensordict.set("ret", ret_normalized)     # [B, T, 1] - 归一化后的
        tensordict.set("state_value", values_normalized)  # [B, T, 1] - 归一化后的（用于 clipping）
        
        # step 5: 最终展平，准备训练
        # 此时数据完全打散，不再有时序概念，等同于 SAC Buffer
        td_flat = tensordict.reshape(-1)

        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            # 全量更新（不使用 Minibatch）
            update_result = self._update(td_flat)
            infos.append(update_result)
        
        if len(infos) == 0: return {}
        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, batch):
        """
        更新函数：
        输入 batch 是一维的 [Minibatch_Size]，完全对照 SAC 风格。
        """
        # 1. 重新提取特征 (带梯度)
        # 输入数据已经是 squeeze 过的扁平数据
        latent = self.shared_features(
            batch["agents", "observation", "camera"],
            batch["agents", "observation", "dynamic_obstacle"],
            batch["agents", "observation", "state"]
        )
        # 将特征注入 batch，供 actor_head 使用
        batch.set("_latent", latent)

        # 2. Actor Update
        # 获取动作分布
        action_dist = self.actor_head.get_dist(batch)
        log_probs = action_dist.log_prob(batch["agents", "action_normalized"])
        entropy_per_dim = action_dist.base_dist.entropy()  # [B, 3] 每个动作维度的熵
        action_entropy = entropy_per_dim.mean()  # 平均而非求和

        # PPO Loss Calculation
        # adv 已经是 [B, 1]，squeeze 成 [B] 进行计算
        # 注意：adv已经在train()中全局归一化，这里直接使用
        advantage = batch["adv"].squeeze(-1)
        
        ratio = torch.exp(log_probs - batch["sample_log_prob"])
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_loss_coefficient * action_entropy

        # 3. Critic Update (with Value Clipping)
        # 获取采样时的旧 value（归一化后的）
        b_value = batch["state_value"].squeeze(-1)  # [B]
        
        # 计算当前策略的新 value
        value = self.critic_head(latent).squeeze(-1)  # [B]
        
        # Value Clipping: 限制 value 更新幅度
        value_clipped = b_value + (value - b_value).clamp(
            -self.cfg.critic.clip_ratio, 
            self.cfg.critic.clip_ratio
        )
        
        # Target: 归一化后的 return
        target = batch["ret"].squeeze(-1)  # [B]
        
        # 计算两种 Critic Loss，取最大值（更保守的更新）
        critic_loss_clipped = self.critic_loss_fn(value_clipped, target)
        critic_loss_original = self.critic_loss_fn(value, target)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # 4. Total Loss & Optimization
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪：分别裁剪和监控 actor 和 critic
        actor_params = [p for p in self.actor_head.parameters() if p.requires_grad]
        critic_params = [p for p in self.critic_head.parameters() if p.requires_grad]
        shared_params = [p for p in self.shared_features.parameters() if p.requires_grad]
        
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, max_norm=5.0)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, max_norm=5.0)
        shared_grad_norm = torch.nn.utils.clip_grad_norm_(shared_params, max_norm=5.0)
        
        self.optimizer.step()

        # 计算 Explained Variance（衡量 Critic 预测质量）
        explained_var = 1 - F.mse_loss(value, target) / target.var()

        return TensorDict({
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": action_entropy.detach(),
            "total_loss": total_loss.detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "shared_grad_norm": shared_grad_norm.detach(),
            "explained_var": explained_var.detach(),
        }, [])