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
import timm



# ==========================================
# 2. 共享特征提取器 (纯净模型，不含维度处理)
# ==========================================
class SharedFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # --- A. ViT Backbone (DINO Pretrained & Frozen) ---
        print("Loading ViT backbone (DINO pretrained)...")
        self.vit = timm.create_model(
            'hf_hub:timm/vit_base_patch16_224.dino',
            pretrained=True
        ).to(device)
        
        # 强制冻结参数
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval() # 始终保持 Eval 模式 (关闭 Dropout/BatchNorm更新)
        
        # 图像预处理参数
        data_config = timm.data.resolve_model_data_config(self.vit)
        self.register_buffer("mean", torch.tensor(data_config['mean']).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor(data_config['std']).view(1, 3, 1, 1).to(device))
        
        # --- B. 动态障碍物提取器 ---
        self.dyn_ext = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            nn.Linear(50, 128), nn.LeakyReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.LeakyReLU(), nn.LayerNorm(64),
        ).to(device)

        # --- C. 融合 MLP (漏斗结构) ---
        # Input: 768(ViT) + 64(Dyn) + 8(State) = 840
        self.fusion_mlp = nn.Sequential(
            nn.Linear(840, 512), nn.LeakyReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.LeakyReLU(), nn.LayerNorm(128)
        ).to(device)

    def forward(self, camera, dynamic_obstacle, state):
        """
        Input shapes are assumed to be flattened: [Batch, ...] 
        No dimension checks inside the model.
        """
        # 1. Image Processing
        x = (camera / 255.0 if camera.max() > 1.1 else camera)
        x = (x - self.mean) / self.std
        
        with torch.no_grad():
            # ViT forward, extract [CLS] token
            v_feat = self.vit.forward_features(x)[:, 0, :] # [Batch, 768]
        
        # 2. Dynamic Obstacle Processing
        d_feat = self.dyn_ext(dynamic_obstacle) # [Batch, 64]
        
        # 3. Concatenation & Fusion
        # state: [Batch, 8]
        combined = torch.cat([v_feat, d_feat, state], dim=-1) # [Batch, 840]
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
        # 统一优化器：包含 Shared MLP, Actor Head, Critic Head
        # 注意：ViT 参数已被冻结，不会包含在梯度更新中
        all_params = list(self.shared_features.parameters()) + \
                     list(self.actor_head.parameters()) + \
                     list(self.critic_head.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=cfg.actor.learning_rate)
        self.gae = GAE(0.99, 0.95)
        self.value_norm = ValueNorm(1).to(device)
        self.critic_loss_fn = nn.HuberLoss(delta=10)

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
        # 注意：这里 tensordict 可能是 [num_env] 维度
        # 为了适应 shared_features，如果维度不对可能需要 view，
        # 但通常 collect 阶段 env 会自动处理好 dimensions.
        
        # 1. 提取特征
        latent = self.shared_features(
            tensordict["agents", "observation", "camera"],
            tensordict["agents", "observation", "dynamic_obstacle"],
            tensordict["agents", "observation", "state"]
        )
        tensordict.set("_latent", latent)
        
        # 2. 策略采样
        self.actor_head(tensordict)
        
        # 3. 价值估计 (可选，PPO推理时不一定需要)
        value = self.critic_head(latent)
        tensordict.set("state_value", value)

        # 4. 坐标转换 (Local -> World)
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        
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
            # Current Features
            latent = self.shared_features(
                td_flat["agents", "observation", "camera"],
                td_flat["agents", "observation", "dynamic_obstacle"],
                td_flat["agents", "observation", "state"]
            )
            
            # Next Features
            next_latent = self.shared_features(
                next_td_flat["agents", "observation", "camera"],
                next_td_flat["agents", "observation", "dynamic_obstacle"],
                next_td_flat["agents", "observation", "state"]
            )

            # -----------------------------------------------------------------
            # [CRITICAL FIX] 修复 ValueNorm AssertionError
            # -----------------------------------------------------------------
            # 1. 获取 Critic 输出，保持 [N, 1] 形状，不要在这里 squeeze！
            values_flat = self.critic_head(latent)       # Shape: [B*T, 1]
            next_values_flat = self.critic_head(next_latent) # Shape: [B*T, 1]
            
            # 2. 进行反归一化 (ValueNorm 要求输入必须有最后一维 1)
            values_flat = self.value_norm.denormalize(values_flat)
            next_values_flat = self.value_norm.denormalize(next_values_flat)

            # 3. 反归一化后，再压缩维度并还原为 [B, T] 供 GAE 使用
            values = values_flat.squeeze(-1).view(B, T)
            next_values = next_values_flat.squeeze(-1).view(B, T)
            # -----------------------------------------------------------------

        # step 3: 准备 GAE 所需数据
        # 原始数据是 [B, T, 1]，squeeze 掉最后一维变成 [B, T]
        rewards = tensordict["next", "agents", "reward"].squeeze(-1) 
        dones = tensordict["next", "terminated"].float().squeeze(-1)

        # step 4: 计算 GAE
        # 输入全部为 [B, T]，GAE 能正确处理时序（即使 T=1）
        adv, ret = self.gae(rewards, dones, values, next_values)
        
        # Update value normalization
        # 注意：ValueNorm 更新需要扁平的 [N, 1] 数据
        self.value_norm.update(ret.reshape(-1, 1))
        
        # 计算归一化后的 Return 用于 Critic Loss Target
        ret_normalized = self.value_norm.normalize(ret.reshape(-1, 1))
        
        # 将计算结果存回 tensordict
        # 必须 unsqueeze 变回 [B, T, 1] 以匹配 Tensordict 标准
        tensordict.set("adv", adv.unsqueeze(-1)) 
        tensordict.set("ret", ret_normalized.view(B, T, 1)) # 存 Normalized 的 Return
        tensordict.set("state_value", values.unsqueeze(-1)) # 存 Denormalized 的 Value (可选)
        
        # step 5: 最终展平，准备 Minibatch 训练
        # 此时数据完全打散，不再有时序概念，等同于 SAC Buffer
        td_flat = tensordict.reshape(-1)

        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            # 16 个样本太少，直接全量更新，不切分 Minibatch
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
        action_entropy = action_dist.entropy().mean()

        # PPO Loss Calculation
        # adv 已经是 [B, 1]，squeeze 成 [B] 进行计算
        advantage = batch["adv"].squeeze(-1) 
        
        # 标准化 Advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        ratio = torch.exp(log_probs - batch["sample_log_prob"])
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_loss_coefficient * action_entropy

        # 3. Critic Update
        # value: [B, 1] -> squeeze -> [B]
        value = self.critic_head(latent).squeeze(-1)
        # ret: [B, 1] -> squeeze -> [B] (Target)
        target = batch["ret"].squeeze(-1)
        
        # Critic Loss (Huber)
        critic_loss = self.critic_loss_fn(value, target)

        # 4. Total Loss & Optimization
        total_loss = actor_loss +  critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        return TensorDict({
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy": action_entropy.detach(),
            "total_loss": total_loss.detach(),
        }, [])