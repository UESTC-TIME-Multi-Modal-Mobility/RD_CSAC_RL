'''
Author: zdytim zdytim@foxmail.com
Date: 2026-01-05 22:31:48
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-05 22:31:50
FilePath: /NavRL/isaac-training/training/scripts/train_with_wandb.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
NavRL + wandb 集成示例
======================

展示如何在 NavRL 训练中集成 wandb 的模型管理功能。

功能特性：
1. 自动模型版本管理
2. 训练过程中的模型上传
3. 模型注册表管理
4. 丰富的模型元数据
"""

import torch
import wandb
import hydra
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.envs import ParallelEnv

# 导入模型管理器
from models import create_navrl_model, load_pretrained_model

# 导入其他必要模块  
from env import NavigationEnv
from utils import make_batch


class WandbNavRLTrainer:
    """
    集成 wandb 的 NavRL 训练器
    
    特色功能：
    1. 自动模型版本管理
    2. 基于性能的模型上传
    3. 训练指标和模型同步跟踪
    4. 模型注册表集成
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 初始化 wandb
        self._init_wandb()
        
        # 创建环境和模型
        self.env = self._create_environment()
        self.model_manager = self._create_model()
        
        # 上传模型架构到 wandb
        self._log_model_architecture()
        
        # 跟踪最佳性能
        self.best_reward = float('-inf')
        self.best_model_step = 0
        
        print("🎉 WandbNavRLTrainer initialized successfully!")
    
    def _init_wandb(self):
        """初始化wandb"""
        wandb_config = {
            'architecture': 'PPO-ViT',
            'framework': 'NavRL',
            'env_config': dict(self.cfg.env),
            'training_config': dict(self.cfg.actor),
        }
        
        wandb.init(
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            config=wandb_config,
            mode=self.cfg.wandb.mode,
            tags=['navrl', 'ppo-vit', 'model-management']
        )
        
        print(f"🔗 wandb initialized: {wandb.run.name}")
    
    def _create_environment(self):
        """创建训练环境"""
        print("🏗️  Creating environment...")
        
        env_fn = lambda: NavigationEnv(self.cfg)
        env = ParallelEnv(
            num_workers=self.cfg.env.num_envs,
            create_env_fn=env_fn,
            device=self.device
        )
        
        # 记录环境信息到wandb
        wandb.config.update({
            'num_envs': self.cfg.env.num_envs,
            'env_type': 'NavigationEnv'
        })
        
        return env
    
    def _create_model(self):
        """创建或加载模型"""
        checkpoint_path = getattr(self.cfg, 'resume_checkpoint', None)
        
        if checkpoint_path and checkpoint_path != "":
            # 检查是否为wandb artifact路径
            if ":" in checkpoint_path and "/" in checkpoint_path:
                print(f"🔄 Loading from wandb artifact: {checkpoint_path}")
                model_manager = create_navrl_model(
                    cfg=self.cfg,
                    observation_spec=self.env.observation_spec,
                    action_spec=self.env.action_spec,
                    device=self.device
                )
                model_manager.load_from_wandb(checkpoint_path)
            else:
                print(f"🔄 Loading from local checkpoint: {checkpoint_path}")
                model_manager = load_pretrained_model(
                    checkpoint_path=checkpoint_path,
                    cfg=self.cfg,
                    observation_spec=self.env.observation_spec,
                    action_spec=self.env.action_spec,
                    device=self.device
                )
        else:
            print("🆕 Creating new model...")
            model_manager = create_navrl_model(
                cfg=self.cfg,
                observation_spec=self.env.observation_spec,
                action_spec=self.env.action_spec,
                device=self.device
            )
        
        return model_manager
    
    def _log_model_architecture(self):
        """记录模型架构信息到wandb"""
        model_info = self.model_manager.get_model().get_model_info()
        
        # 更新wandb配置
        wandb.config.update({
            'model_info': model_info,
            'total_parameters': model_info['total_parameters'],
            'trainable_parameters': model_info['trainable_parameters']
        })
        
        # 记录模型摘要
        wandb.log({
            'model/total_parameters': model_info['total_parameters'],
            'model/trainable_parameters': model_info['trainable_parameters'],
            'model/frozen_parameters': model_info['frozen_parameters']
        })
        
        print(f"📊 Model info logged to wandb")
    
    def train(self):
        """主训练循环"""
        print("🚀 Starting training with wandb integration...")
        
        model = self.model_manager.get_model()
        step = 0
        
        # 模拟训练循环
        for episode in range(self.cfg.total_episodes):
            # 1. 环境交互和数据收集
            rollout_data = self._collect_rollout(model, episode)
            
            # 2. 模型训练
            train_info = self._train_step(model, rollout_data)
            step += 1
            
            # 3. 记录训练信息
            wandb.log({
                'train/episode': episode,
                'train/step': step,
                **{f'train/{k}': v for k, v in train_info.items()}
            })
            
            # 4. 定期评估和模型保存
            if episode % self.cfg.eval_interval == 0:
                eval_metrics = self._evaluate(model, episode)
                
                # 记录评估指标
                wandb.log({
                    'eval/episode': episode,
                    **{f'eval/{k}': v for k, v in eval_metrics.items()}
                })
                
                # 检查是否需要保存最佳模型
                current_reward = eval_metrics['mean_reward']
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    self.best_model_step = step
                    
                    print(f"🏆 New best model! Reward: {current_reward:.2f}")
                    self._save_best_model(step, eval_metrics)
            
            # 5. 定期保存检查点
            if episode % self.cfg.save_interval == 0:
                self._save_checkpoint(step, train_info)
        
        # 6. 训练结束，上传最终模型到注册表
        self._upload_final_model(step)
        
        print("🎉 Training completed!")
    
    def _collect_rollout(self, model, episode):
        """收集rollout数据"""
        # 模拟数据收集
        td = self.env.reset()
        rollout_data = []
        
        for _ in range(self.cfg.rollout_length):
            td = model(td)
            td = self.env.step(td)
            rollout_data.append(td.clone())
        
        return rollout_data
    
    def _train_step(self, model, rollout_data):
        """执行训练步骤"""
        # 将数据转换为batch
        batch_td = make_batch(rollout_data, self.device)
        
        # 训练模型
        train_info = model.train(batch_td)
        
        return train_info
    
    def _evaluate(self, model, episode):
        """评估模型性能"""
        print(f"🔍 Evaluating model at episode {episode}...")
        
        self.model_manager.set_training_mode(False)
        
        rewards = []
        success_rates = []
        
        # 运行多个评估episode
        for eval_ep in range(10):
            td = self.env.reset()
            episode_reward = 0
            success = False
            
            for _ in range(200):  # 最大步数
                with torch.no_grad():
                    td = model(td)
                    td = self.env.step(td)
                    
                    reward = td["next", "agents", "reward"].sum().item()
                    episode_reward += reward
                    
                    # 检查是否成功（可根据任务定义）
                    if reward > 10:  # 示例成功条件
                        success = True
                    
                    if td["next", "terminated"].any():
                        break
            
            rewards.append(episode_reward)
            success_rates.append(1.0 if success else 0.0)
        
        # 恢复训练模式
        self.model_manager.set_training_mode(True)
        
        eval_metrics = {
            'mean_reward': sum(rewards) / len(rewards),
            'std_reward': torch.tensor(rewards).std().item(),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'success_rate': sum(success_rates) / len(success_rates)
        }
        
        print(f"   📊 Evaluation results: {eval_metrics}")
        return eval_metrics
    
    def _save_best_model(self, step, eval_metrics):
        """保存最佳模型"""
        save_path = f"models/best_model_step_{step}.pt"
        
        # 保存并上传到wandb，带有'best'别名
        self.model_manager.save_checkpoint(
            filepath=save_path,
            step=step,
            additional_info={
                'eval_metrics': eval_metrics,
                'best_reward': self.best_reward,
                'model_type': 'best'
            },
            upload_to_wandb=True,
            wandb_alias='best'
        )
        
        print(f"💎 Best model saved and uploaded: {save_path}")
    
    def _save_checkpoint(self, step, train_info):
        """保存常规检查点"""
        save_path = f"models/checkpoint_step_{step}.pt"
        
        # 保存并可选上传到wandb
        upload_to_wandb = step % (self.cfg.save_interval * 5) == 0  # 每5次保存上传一次
        
        self.model_manager.save_checkpoint(
            filepath=save_path,
            step=step,
            additional_info={
                'train_info': train_info,
                'model_type': 'checkpoint'
            },
            upload_to_wandb=upload_to_wandb,
            wandb_alias='latest' if upload_to_wandb else None
        )
        
        if upload_to_wandb:
            print(f"☁️  Checkpoint uploaded to wandb: step {step}")
    
    def _upload_final_model(self, step):
        """上传最终模型到注册表"""
        print("🎯 Uploading final model to wandb registry...")
        
        self.model_manager.upload_model_to_registry(
            model_name=f"navrl-final-{wandb.run.id}",
            description=f"Final NavRL model trained for {step} steps with best reward {self.best_reward:.2f}",
            tags=['final', 'production-ready', f'reward-{self.best_reward:.1f}'],
            step=step
        )


# === 使用示例 ===

@hydra.main(version_base=None, config_path="../cfg", config_name="train_ppo_vit")
def main(cfg: DictConfig):
    """
    主训练函数 - 使用wandb模型管理
    
    示例配置：
    ```yaml
    wandb:
      project: "navrl-models"
      name: "ppo-vit-experiment"
      mode: "online"  # 或 "offline", "disabled"
    
    # 从wandb artifact恢复训练
    resume_checkpoint: "username/navrl-models/navrl-model-step-1000:best"
    
    # 或从本地检查点恢复
    # resume_checkpoint: "path/to/checkpoint.pt"
    ```
    """
    print("🎯 NavRL Training with wandb Model Management")
    print("=" * 60)
    
    # 创建训练器
    trainer = WandbNavRLTrainer(cfg)
    
    # 开始训练
    trainer.train()
    
    # 结束wandb运行
    wandb.finish()
    print("🎉 Training session completed!")


# === 高级用法示例 ===

def download_and_evaluate_model():
    """
    下载wandb模型并进行评估的示例
    """
    print("📥 Downloading model from wandb for evaluation...")
    
    # 创建模型管理器
    model_manager = create_navrl_model(cfg, obs_spec, act_spec, device)
    
    # 从wandb下载模型
    success = model_manager.load_from_wandb("username/project/model-name:best")
    
    if success:
        # 进行评估
        model = model_manager.get_model()
        # ... 评估代码 ...
        print("✅ Model evaluation completed")
    else:
        print("❌ Failed to load model from wandb")


def compare_model_versions():
    """
    比较不同版本模型性能的示例
    """
    print("🔍 Comparing model versions...")
    
    versions = ["v1", "v2", "best", "latest"]
    results = {}
    
    for version in versions:
        model_manager = create_navrl_model(cfg, obs_spec, act_spec, device)
        success = model_manager.load_from_wandb(f"project/model-name:{version}")
        
        if success:
            # 评估模型
            eval_result = evaluate_model(model_manager.get_model())
            results[version] = eval_result
            print(f"   {version}: {eval_result['mean_reward']:.2f}")
    
    # 找到最佳版本
    best_version = max(results.keys(), key=lambda v: results[v]['mean_reward'])
    print(f"🏆 Best version: {best_version}")


if __name__ == "__main__":
    main()


# === wandb 模型管理功能总结 ===
"""
🎯 NavRL + wandb 模型管理功能：

1. **自动模型上传**：
   - 训练过程中自动保存和上传模型
   - 支持版本别名（best, latest, v1, v2等）
   - 丰富的模型元数据和模型卡片

2. **模型注册表**：
   - 生产级模型版本管理
   - 模型生命周期跟踪
   - 标签和描述管理

3. **性能追踪**：
   - 训练指标和模型版本关联
   - 最佳模型自动识别和保存
   - 评估指标持续跟踪

4. **便捷加载**：
   - 支持从wandb artifact直接加载模型
   - 版本比较和性能对比
   - 跨实验模型共享

使用命令：
```bash
# 基础训练
python train_with_wandb.py

# 从wandb artifact恢复训练
python train_with_wandb.py resume_checkpoint="user/project/model:best"

# 离线模式训练
python train_with_wandb.py wandb.mode=offline
```

优势：
✅ 完整的模型生命周期管理
✅ 自动版本控制和元数据跟踪  
✅ 便捷的模型分享和协作
✅ 集成训练指标和模型性能
✅ 生产环境模型部署支持
"""