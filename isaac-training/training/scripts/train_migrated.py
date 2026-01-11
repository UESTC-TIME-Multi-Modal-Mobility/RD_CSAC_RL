'''
Author: zdytim zdytim@foxmail.com
Date: 2026-01-05 22:21:54
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-05 22:21:55
FilePath: /NavRL/isaac-training/training/scripts/train_migrated.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
原始训练脚本的迁移示例
========================

展示如何将现有的ppo_vit_v3.py训练逻辑迁移到新的模型管理器。

主要更改：
1. 导入新的模型管理器
2. 替换模型创建逻辑
3. 使用统一的检查点管理
4. 保持原有的训练循环不变
"""

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.envs import ParallelEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# 导入新的模型管理器 - 这是主要改动
from models import create_navrl_model, load_pretrained_model

# 其他导入保持不变
from env import NavigationEnv
from utils import make_batch, vec_to_world


def create_environment(cfg, device):
    """创建环境 - 保持原有逻辑"""
    print("🏗️  Creating environment...")
    
    env_fn = lambda: NavigationEnv(cfg)
    env = ParallelEnv(
        num_workers=cfg.env.num_envs,
        create_env_fn=env_fn,
        device=device
    )
    
    print(f"   ✅ Created with {cfg.env.num_envs} parallel environments")
    return env


def create_collector(env, model, cfg, device):
    """创建数据收集器 - 保持原有逻辑"""
    print("📊 Creating data collector...")
    
    # 创建收集器
    collector = SyncDataCollector(
        env,
        model,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        storing_device=device,
    )
    
    print(f"   ✅ Collector ready: {cfg.frames_per_batch} frames/batch")
    return collector


def create_replay_buffer(cfg, device):
    """创建经验回放缓冲区 - 保持原有逻辑"""
    if not cfg.use_replay_buffer:
        return None
    
    print("💾 Creating replay buffer...")
    
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.buffer_size, device=device),
        batch_size=cfg.minibatch_size,
        pin_memory=False,
        prefetch=3,
    )
    
    print(f"   ✅ Buffer ready: {cfg.buffer_size} capacity")
    return buffer


@hydra.main(version_base=None, config_path="../cfg", config_name="train_ppo_vit")
def main(cfg: DictConfig):
    """
    主训练函数 - 使用新的模型管理器
    
    主要更改：
    1. 使用create_navrl_model()替代直接实例化PPOVIT
    2. 通过model_manager管理模型状态
    3. 统一的检查点加载/保存
    """
    print("🚀 NavRL Training with New Model Manager")
    print("=" * 60)
    
    # 设备配置
    device = torch.device(cfg.device)
    print(f"🔧 Using device: {device}")
    
    # 1. 创建环境
    env = create_environment(cfg, device)
    observation_spec = env.observation_spec
    action_spec = env.action_spec
    
    # 2. 创建模型管理器 - 主要改动在这里
    print("🧠 Creating model...")
    
    # 检查是否需要从检查点恢复
    resume_checkpoint = getattr(cfg, 'resume_checkpoint', None)
    
    if resume_checkpoint and resume_checkpoint != "":
        # 加载预训练模型
        model_manager = load_pretrained_model(
            checkpoint_path=resume_checkpoint,
            cfg=cfg,
            observation_spec=observation_spec,
            action_spec=action_spec,
            device=device,
            load_optimizer=True
        )
    else:
        # 创建新模型
        model_manager = create_navrl_model(
            cfg=cfg,
            observation_spec=observation_spec,
            action_spec=action_spec,
            device=device
        )
    
    # 获取模型实例
    model = model_manager.get_model()
    
    # 打印模型摘要
    model_manager.print_model_summary()
    
    # 3. 创建数据收集器
    collector = create_collector(env, model, cfg, device)
    
    # 4. 创建经验回放缓冲区（可选）
    replay_buffer = create_replay_buffer(cfg, device)
    
    # 5. 训练循环 - 保持原有逻辑
    print("🏃 Starting training loop...")
    
    collected_frames = 0
    for i, data in enumerate(collector):
        # 训练步骤计数
        current_frames = data.numel()
        collected_frames += current_frames
        
        print(f"\nStep {i+1} - Frames: {current_frames} (Total: {collected_frames})")
        
        # 设置训练模式
        model_manager.set_training_mode(True)
        
        # 训练更新 - 使用模型的train方法
        with torch.cuda.amp.autocast(enabled=getattr(cfg, 'use_amp', True)):
            train_info = model.train(data)
        
        # 记录训练信息
        if i % cfg.log_interval == 0:
            print("📊 Training metrics:")
            for key, value in train_info.items():
                print(f"   {key}: {value:.4f}")
        
        # 可选：添加到经验回放缓冲区
        if replay_buffer is not None:
            replay_buffer.extend(data.reshape(-1))
        
        # 保存检查点
        if i > 0 and i % cfg.save_interval == 0:
            save_path = f"outputs/checkpoint_step_{collected_frames}.pt"
            model_manager.save_checkpoint(
                filepath=save_path,
                step=collected_frames,
                additional_info={
                    'training_step': i,
                    'collected_frames': collected_frames,
                    'train_info': train_info
                }
            )
            print(f"💾 Checkpoint saved: {save_path}")
        
        # 检查是否达到总帧数
        if collected_frames >= cfg.total_frames:
            print(f"🎉 Training completed! Total frames: {collected_frames}")
            break
    
    # 6. 最终评估
    print("\n🔍 Final evaluation...")
    model_manager.set_training_mode(False)
    
    # 简单评估循环
    eval_rewards = []
    for eval_episode in range(10):
        td = env.reset()
        episode_reward = 0
        
        for _ in range(200):  # 最大步数
            with torch.no_grad():
                td = model(td)
                td = env.step(td)
                
                reward = td["next", "agents", "reward"].sum().item()
                episode_reward += reward
                
                if td["next", "terminated"].any():
                    break
        
        eval_rewards.append(episode_reward)
    
    avg_reward = sum(eval_rewards) / len(eval_rewards)
    print(f"📊 Average evaluation reward: {avg_reward:.2f}")
    
    # 保存最终模型
    final_save_path = "outputs/final_model.pt"
    model_manager.save_checkpoint(
        filepath=final_save_path,
        step=collected_frames,
        additional_info={
            'final_eval_reward': avg_reward,
            'eval_rewards': eval_rewards,
            'training_completed': True
        }
    )
    
    print(f"💾 Final model saved: {final_save_path}")
    print("🎉 Training session completed successfully!")


# === 迁移指南 ===
"""
从ppo_vit_v3.py迁移到新模型管理器的步骤：

1. 导入更改：
   OLD: from ppo_vit_v3 import PPOVIT
   NEW: from models import create_navrl_model, load_pretrained_model

2. 模型创建：
   OLD: model = PPOVIT(cfg, observation_spec, action_spec, device)
   NEW: model_manager = create_navrl_model(cfg, observation_spec, action_spec, device)
        model = model_manager.get_model()

3. 检查点加载：
   OLD: model.load_full_checkpoint(checkpoint_path)
   NEW: model_manager = load_pretrained_model(checkpoint_path, cfg, obs_spec, act_spec, device)

4. 检查点保存：
   OLD: torch.save({'model_state_dict': model.state_dict(), ...}, path)
   NEW: model_manager.save_checkpoint(path, step=step, additional_info={...})

5. 参数管理：
   OLD: model.shared_features.freeze_vit_encoder()
   NEW: model_manager.freeze_vit_encoder()

优势：
- ✅ 更清晰的代码结构
- ✅ 统一的接口
- ✅ 更好的错误处理
- ✅ 自动的参数统计和管理
- ✅ 灵活的配置选项
"""


if __name__ == "__main__":
    main()