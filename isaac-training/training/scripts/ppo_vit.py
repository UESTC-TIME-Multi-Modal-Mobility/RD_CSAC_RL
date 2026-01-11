'''
Author: zdytim zdytim@foxmail.com
Date: 2025-12-20 16:59:18
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2025-12-21 18:03:36
FilePath: /NavRL/isaac-training/training/scripts/ppo copy.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
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
from model import ViT,ConvNet
import timm


class PPOVIT(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # 从observation_spec获取实际的batch维度信息
        dummy_input = observation_spec.zero()
        self.original_batch_size = dummy_input.batch_size
        
        # --- 用 ViT 替换 lidar CNN ---
        print("Loading ViT backbone (DINO pretrained)...")
        self.vit_model = timm.create_model(
            'hf_hub:timm/vit_base_patch16_224.dino',
            pretrained=True
        ).to(self.device)
        for param in self.vit_model.parameters():
            param.requires_grad = False
        self.vit_model.eval()
        
        data_config = timm.data.resolve_model_data_config(self.vit_model)
        self.mean = torch.tensor(data_config['mean']).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(data_config['std']).view(1, 3, 1, 1).to(self.device)

        # 智能特征提取器，自动处理batch维度
        class SmartViTExtractor(nn.Module):
            def __init__(self, vit_model, mean, std):
                super().__init__()
                self.vit_model = vit_model
                self.mean = mean
                self.std = std
                self.proj = nn.Linear(768, 128)
                
            def forward(self, x):
                orig_shape = x.shape
                # 自动处理不同的batch维度
                if x.dim() == 5:  # [B, T, C, H, W]
                    B, T = x.shape[:2]
                    x = x.reshape(-1, *x.shape[2:])
                    restore_shape = True
                else:  # [B, C, H, W]
                    restore_shape = False
                    
                if x.numel() == 0:
                    return torch.empty(0, 128, device=x.device, dtype=x.dtype)
                    
                if x.max() > 1.1:
                    x = x / 255.0
                x = (x - self.mean) / self.std
                
                with torch.no_grad():
                    feats = self.vit_model.forward_features(x)
                    feats = feats[:, 0, :]  # [batch, 768]
                feats = self.proj(feats)  # [batch, 128]
                
                if restore_shape:
                    feats = feats.view(B, T, -1)
                    
                return feats

        class SmartDynamicObstacleExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.rearrange = Rearrange("n c w h -> n (c w h)")
                self.mlp = make_mlp([50, 128, 64])  # 根据实际输入维度调整

            def forward(self, x):
                orig_shape = x.shape
                if x.dim() == 5:  # [B, T, C, H, W]
                    B, T = x.shape[:2]
                    x = x.reshape(-1, *x.shape[2:])
                    restore_shape = True
                else:
                    restore_shape = False
                    
                if x.numel() == 0:
                    return torch.empty(0, 64, device=x.device, dtype=x.dtype)
                    
                x = self.rearrange(x)
                x = self.mlp(x)
                
                if restore_shape:
                    x = x.view(B, T, -1)
                    
                return x

        class SmartFlattenFeature(nn.Module):
            def forward(self, x):
                # 保持batch维度，只flatten特征维度
                if x.dim() == 2:  # [B, features] - 已经是正确形状
                    return x
                elif x.dim() == 3:  # [B, T, features] - 只在特征维度flatten
                    output = x.view(x.shape[0], -1)  # [B, T*features]
                    return output
                else:  # [B, ...] 多维特征 - flatten除batch外的所有维度
                    output = x.view(x.shape[0], -1)  # [B, flattened_features]
                    return output

        # 初始化网络组件
        self.vit_extractor = SmartViTExtractor(self.vit_model, self.mean, self.std)
        self.dynamic_extractor = SmartDynamicObstacleExtractor()
        self.flatten_feature = SmartFlattenFeature()

        # 构建完整的特征提取器
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(self.vit_extractor, [("agents", "observation", "camera")], ["_vit_feature"]),
            TensorDictModule(self.dynamic_extractor, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            TensorDictModule(self.flatten_feature, ["_vit_feature"], ["_vit_feature_flat"]),
            TensorDictModule(self.flatten_feature, [("agents", "observation", "state")], ["_state_flat"]),
            TensorDictModule(self.flatten_feature, ["_dynamic_obstacle_feature"], ["_dynamic_obstacle_feature_flat"]),
            CatTensors(["_vit_feature_flat", "_state_flat", "_dynamic_obstacle_feature_flat"], "_feature", del_keys=False),
            TensorDictModule(make_mlp([200, 256]), ["_feature"], ["_feature"]),  # 128+8+64=200
        ).to(self.device)

        # Actor network
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.Linear(256, 1), ["_feature"], ["state_value"]  # 不用LazyLinear
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        # 统一优化器 - 不拆分feature_extractor
        all_params = list(self.feature_extractor.parameters()) + \
                    list(self.actor.parameters()) + \
                    list(self.critic.parameters())
        
        self.optimizer = torch.optim.Adam(all_params, lr=cfg.actor.learning_rate)

        # 智能初始化 - 自动适配batch维度
        self._initialize_with_dummy_input(observation_spec)

        # Initialize network weights
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    def _initialize_with_dummy_input(self, observation_spec):
        """智能初始化，自动处理batch维度差异"""
        dummy_input = observation_spec.zero()
        
        # 检查是否需要添加时间维度
        try:
            sample_obs = dummy_input["agents", "observation", "camera"]
            
            if sample_obs.dim() == 4:  # [B, C, H, W]，需要添加时间维度
                T = 1  # 使用更小的时间步来避免内存问题
                
                # 安全地获取观测字典的keys
                obs_dict = dummy_input["agents", "observation"]
                if hasattr(obs_dict, 'keys'):
                    keys_to_process = list(obs_dict.keys())
                else:
                    # 如果无法获取keys，手动指定主要的observation keys
                    keys_to_process = ["camera", "state", "dynamic_obstacle", "lidar", "direction"]
                
                for key in keys_to_process:
                    try:
                        if ("agents", "observation", key) not in dummy_input.keys(True):
                            continue
                            
                        value = dummy_input["agents", "observation", key]
                        
                        if value.dim() == 4:  # camera等图像数据
                            new_value = value.unsqueeze(1)  # [B, 1, C, H, W]
                            dummy_input["agents", "observation", key] = new_value
                        elif key == "state" and value.dim() == 2:  # state数据
                            new_value = value.unsqueeze(1)  # [B, 1, features]
                            dummy_input["agents", "observation", key] = new_value
                        elif value.dim() == 3 and key in ["direction"]:  # 3维数据
                            new_value = value.unsqueeze(1)  # [B, 1, H, W]
                            dummy_input["agents", "observation", key] = new_value
                    except (KeyError, IndexError) as e:
                        continue
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not find camera observation for initialization: {e}")
        
        # 执行一次前向传播来初始化所有参数
        with torch.no_grad():
            self(dummy_input)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Coordinate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    def _process_tensordict_batch(self, tensordict, batch_size=32):
        """智能批处理，自动处理不同的batch维度"""
        # 获取原始batch信息
        original_shape = tensordict.batch_size
        
        if len(original_shape) == 2:  # [B, T]
            B, T = original_shape
            tensordict = tensordict.reshape(-1)  # 展平为[B*T]
            total = B * T
            restore_shape = (B, T)
        else:  # [B]
            total = original_shape[0]
            restore_shape = None
            
        # 分批处理
        td_list = []
        for i in range(0, total, batch_size):
            td_chunk = tensordict[i:i+batch_size]
            if td_chunk.batch_size[0] > 0:  # 确保batch不为空
                out_chunk = self.feature_extractor(td_chunk)
                td_list.append(out_chunk)
        
        if td_list:
            result = torch.cat(td_list, 0)
        else:
            result = tensordict  # 如果没有有效batch，返回原始tensordict
            
        return result, restore_shape

    def train(self, tensordict):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        next_tensordict = tensordict["next"]
        
        with torch.no_grad():
            # 智能处理next_tensordict的批维度
            next_tensordict, restore_shape = self._process_tensordict_batch(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
            
        # 获取rewards, dones, values
        rewards = tensordict["next", "agents", "reward"]
        dones = tensordict["next", "terminated"]
        values = tensordict["state_value"]
        
        # Denormalize values
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)
        
        # 智能reshape为GAE所需的[B, T]格式
        if restore_shape is not None:
            B, T = restore_shape
            rewards = rewards.view(B, T)
            dones = dones.view(B, T)
            values = values.view(B, T)
            next_values = next_values.view(B, T)
        else:
            # 如果原本就是一维batch，需要添加时间维度
            B = rewards.shape[0]
            T = 1
            rewards = rewards.view(B, T)
            dones = dones.view(B, T)
            values = values.view(B, T)
            next_values = next_values.view(B, T)
            
        # Calculate GAE
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        
        # Update value normalization
        ret_flat = ret.view(-1, 1)
        self.value_norm.update(ret_flat)
        ret = self.value_norm.normalize(ret_flat).view_as(ret)
        
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        # Training loop
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                update_result = self._update(minibatch)
                infos.append(update_result)
                    
        if len(infos) == 0:
            # 如果所有批次都是空的，返回零损失
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "entropy_loss": 0.0,
                "total_loss": 0.0
            }
                
        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict):
        """统一的更新函数，不分离feature_extractor训练"""
        # 检查是否为空批次
        if tensordict.batch_size[0] == 0:
            print("Warning: Skipping empty minibatch")
            # 返回与正常情况相同结构的零值TensorDict
            return TensorDict({
                "actor_loss": torch.tensor(0.0),
                "critic_loss": torch.tensor(0.0), 
                "entropy_loss": torch.tensor(0.0),
                "total_loss": torch.tensor(0.0),
                "grad_norm": torch.tensor(0.0),
                "explained_var": torch.tensor(0.0)
            }, [])
            
        self.feature_extractor(tensordict)

        # Get action distribution
        action_dist = self.actor.get_dist(tensordict)
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")])

        # Entropy Loss
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # Actor Loss (PPO)
        advantage = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

        # Critic Loss
        b_value = tensordict["state_value"]
        ret = tensordict["ret"]
        value = self.critic(tensordict)["state_value"] 
        
        # 确保形状匹配：将 ret 扩展为与 value 相同的形状
        if ret.dim() != value.dim():
            ret = ret.unsqueeze(-1)  # [batch] -> [batch, 1]
        
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio)
        critic_loss_clipped = self.critic_loss_fn(value_clipped, ret)
        critic_loss_original = self.critic_loss_fn(value, ret)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # Total Loss
        total_loss = entropy_loss + actor_loss + critic_loss

        # 统一优化 - 一个optimizer搞定所有参数
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping

        
        self.optimizer.step()


        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy_loss": entropy_loss,
            "total_loss": total_loss,
        }, [])