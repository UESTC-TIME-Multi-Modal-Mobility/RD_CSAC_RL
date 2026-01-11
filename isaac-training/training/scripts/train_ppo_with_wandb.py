'''
Author: zdytim zdytim@foxmail.com
Date: 2026-01-05 22:33:38
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-05 22:33:39
FilePath: /NavRL/isaac-training/training/scripts/train_ppo_with_wandb.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
NavRL 训练脚本 - 集成 wandb 模型管理
====================================

在原有的 train_ppo.py 基础上集成 wandb 模型上传功能。
主要增加：
1. 自动模型版本管理
2. 基于性能的最佳模型保存
3. 模型元数据跟踪
4. 便捷的模型恢复功能
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import datetime
import wandb

# 导入原有模块
from env import NavigationEnv
from ppo_vit_v3 import PPOVIT  # 或者使用新的模型管理器
from utils import make_eval_env

# 导入 wandb 模型工具
from wandb_model_utils import (
    upload_model_to_wandb, 
    save_and_upload_best_model,
    download_model_from_wandb,
    log_model_metrics
)


@hydra.main(version_base=None, config_path="../cfg", config_name="train_ppo_vit")
def main(cfg: DictConfig):
    """
    主训练函数 - 集成 wandb 模型管理
    """
    print("🚀 NavRL Training with wandb Model Management")
    print("=" * 60)
    
    # === 1. 环境和设备设置 ===
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # === 2. wandb 初始化 ===
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    if cfg.wandb.run_id is None:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name + "/" + datetime.datetime.now().strftime('%m-%d_%H-%M'),
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
            tags=['navrl', 'ppo-vit', 'model-management']  # 添加标签
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name + "/" + datetime.datetime.now().strftime('%m-%d_%H-%M'),
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )
    
    print(f"🔗 wandb run: {run.name}")
    
    # === 3. 环境创建 ===
    print("🏗️  Creating environments...")
    env = NavigationEnv(cfg)
    eval_env = make_eval_env(cfg)
    
    # === 4. 模型创建 ===
    print("🧠 Creating model...")
    observation_spec = env.observation_spec
    action_spec = env.action_spec
    
    policy = PPOVIT(cfg, observation_spec, action_spec, device)
    
    # 记录模型架构信息到 wandb
    model_info = {
        'total_parameters': sum(p.numel() for p in policy.parameters()),
        'trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad),
        'architecture': 'PPO-ViT',
        'action_dim': policy.action_dim
    }
    
    wandb.config.update({'model_info': model_info})
    print(f"📊 Model parameters: {model_info['total_parameters']:,}")
    
    # === 5. 模型恢复（可选）===
    resume_checkpoint = getattr(cfg, 'resume_checkpoint', None)
    if resume_checkpoint:
        if ":" in resume_checkpoint and "/" in resume_checkpoint:
            # 从 wandb artifact 恢复
            print(f"📥 Resuming from wandb artifact: {resume_checkpoint}")
            model_path = download_model_from_wandb(resume_checkpoint)
            if model_path:
                policy.load_state_dict(torch.load(model_path, map_location=device))
                print("✅ Model restored from wandb")
        else:
            # 从本地文件恢复
            print(f"📥 Resuming from local checkpoint: {resume_checkpoint}")
            if os.path.exists(resume_checkpoint):
                policy.load_state_dict(torch.load(resume_checkpoint, map_location=device))
                print("✅ Model restored from local file")
    
    # === 6. 最佳模型跟踪器 ===
    best_model_tracker = {
        'best_value': float('-inf'),
        'best_step': 0,
        'threshold_metric': 'mean_reward'
    }
    
    # === 7. 主训练循环 ===
    print("🏃 Starting training loop...")
    
    collector = make_collector(env, policy, cfg, device)  # 假设这个函数存在
    
    for i, data in enumerate(collector):
        current_frames = data.numel()
        
        print(f"\\nStep {i+1} - Frames: {current_frames}")
        
        # 训练更新
        with torch.cuda.amp.autocast(enabled=getattr(cfg, 'use_amp', True)):
            train_info = policy.train(data)
        
        # 基础指标记录
        info = {f'train/{k}': v for k, v in train_info.items()}
        
        # === 定期评估 ===
        if i % cfg.eval_interval == 0:
            print(f"🔍 Evaluating at step {i}...")
            
            eval_info = evaluate_model(eval_env, policy, cfg)
            info.update({f'eval/{k}': v for k, v in eval_info.items()})
            
            # 记录模型相关指标
            log_model_metrics(eval_info, step=i)
            
            # 检查并保存最佳模型
            is_new_best = save_and_upload_best_model(
                model_state_dict=policy.state_dict(),
                step=i,
                eval_metrics=eval_info,
                threshold_metric=best_model_tracker['threshold_metric'],
                best_value_tracker=best_model_tracker
            )
            
            if is_new_best:
                print(f"🏆 New best model saved! {best_model_tracker['threshold_metric']}: {best_model_tracker['best_value']:.4f}")
            
            print(f"📊 Evaluation completed.")
        
        # 记录所有指标到 wandb
        run.log(info, step=i)
        
        # === 定期保存检查点 ===
        if i % cfg.save_interval == 0:
            # 本地保存
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print(f"💾 Checkpoint saved locally: checkpoint_{i}.pt")
            
            # 上传到 wandb（每5次保存上传一次，避免过于频繁）
            if i % (cfg.save_interval * 5) == 0:
                upload_model_to_wandb(
                    model_state_dict=policy.state_dict(),
                    step=i,
                    eval_metrics=info.get('eval', {}),
                    model_alias="latest",
                    model_type="checkpoint"
                )
        
        # === 训练摘要打印 ===
        if i % 100 == 0:
            print(f"\\n[Step {i}] Training Summary:")
            for k, v in info.items():
                if isinstance(v, (float, int)):
                    print(f"  {k:<20}: {v:.4f}")
            print("-" * 40)
        
        # 检查训练结束条件
        if i >= cfg.total_training_steps:
            break
    
    # === 8. 训练结束处理 ===
    print("🎯 Training completed! Saving final model...")
    
    # 保存最终模型
    final_ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), final_ckpt_path)
    
    # 上传最终模型到 wandb，带有特殊标记
    final_eval_metrics = evaluate_model(eval_env, policy, cfg)
    upload_model_to_wandb(
        model_state_dict=policy.state_dict(),
        step=i,
        eval_metrics=final_eval_metrics,
        model_alias="final",
        model_type="final"
    )
    
    # 创建训练摘要
    training_summary = {
        'total_steps': i,
        'best_model_step': best_model_tracker['best_step'],
        'best_model_reward': best_model_tracker['best_value'],
        'final_eval_reward': final_eval_metrics.get('mean_reward', 0),
        'total_parameters': model_info['total_parameters']
    }
    
    # 记录训练摘要
    wandb.log({"training_summary": training_summary})
    
    print("🎉 Training session completed successfully!")
    print(f"   📊 Best model: step {best_model_tracker['best_step']}, reward {best_model_tracker['best_value']:.2f}")
    print(f"   📈 Final reward: {final_eval_metrics.get('mean_reward', 0):.2f}")
    
    # 结束 wandb 运行
    wandb.finish()
    
    # 关闭环境
    if hasattr(env, 'close'):
        env.close()
    if hasattr(eval_env, 'close'):
        eval_env.close()


def evaluate_model(eval_env, policy, cfg, num_episodes=10):
    """
    评估模型性能
    
    Args:
        eval_env: 评估环境
        policy: 策略模型
        cfg: 配置
        num_episodes: 评估回合数
        
    Returns:
        评估指标字典
    """
    print(f"   🔍 Running {num_episodes} evaluation episodes...")
    
    policy.eval()
    rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        td = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        
        with torch.no_grad():
            for step in range(cfg.max_episode_length):
                td = policy(td)
                td = eval_env.step(td)
                
                reward = td["next", "agents", "reward"].sum().item()
                episode_reward += reward
                episode_length += 1
                
                # 检查成功条件（根据你的任务定义）
                if reward > cfg.success_reward_threshold:  # 假设配置中有这个阈值
                    success_count += 1
                
                if td["next", "terminated"].any():
                    break
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    policy.train()
    
    # 计算统计指标
    eval_metrics = {
        'mean_reward': sum(rewards) / len(rewards),
        'std_reward': torch.tensor(rewards).std().item(),
        'min_reward': min(rewards),
        'max_reward': max(rewards),
        'mean_episode_length': sum(episode_lengths) / len(episode_lengths),
        'success_rate': success_count / num_episodes
    }
    
    return eval_metrics


def make_collector(env, policy, cfg, device):
    """
    创建数据收集器
    这里需要根据你的具体实现来调整
    """
    # 这是一个占位函数，需要根据你的实际收集器实现来填写
    from torchrl.collectors import SyncDataCollector
    
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        storing_device=device,
    )
    
    return collector


# === 配置示例 ===
"""
使用示例配置 (cfg/train_ppo_vit.yaml):

```yaml
# 基础配置
device: "cuda:0"
total_training_steps: 10000
eval_interval: 100
save_interval: 200

# 评估配置
max_episode_length: 1000
success_reward_threshold: 50.0

# wandb 配置
wandb:
  project: "navrl-models"
  name: "ppo-vit-experiment"
  mode: "online"

# 模型恢复 (可选)
# resume_checkpoint: "path/to/checkpoint.pt"  # 本地文件
# resume_checkpoint: "username/project/model:best"  # wandb artifact

# 其他原有配置...
```

运行命令：
```bash
# 基础训练
python train_ppo_with_wandb.py

# 从 wandb artifact 恢复
python train_ppo_with_wandb.py resume_checkpoint="user/project/model:best"

# 离线模式
python train_ppo_with_wandb.py wandb.mode=offline
```
"""


if __name__ == "__main__":
    main()