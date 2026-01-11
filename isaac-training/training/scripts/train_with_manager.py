'''
Author: zdytim zdytim@foxmail.com
Date: 2026-01-05 22:21:12
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-05 22:21:13
FilePath: /NavRL/isaac-training/training/scripts/train_with_manager.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
NavRL训练脚本示例 - 使用新的模型管理器
=============================================

展示如何使用抽象的NavRL模型管理器来简化训练代码。

主要改进：
1. 模型创建和管理逻辑分离
2. 统一的检查点加载/保存接口
3. 清晰的参数管理和配置
4. 更好的代码组织和复用性

使用方法：
1. 导入模型管理器
2. 创建或加载模型  
3. 使用统一的训练接口
"""

import torch
import hydra
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.envs import ParallelEnv

# 导入新的模型管理器
from models import create_navrl_model, load_pretrained_model

# 导入其他必要模块
from env import NavigationEnv
from utils import make_batch, vec_to_world


class NavRLTrainer:
    """
    NavRL训练器 - 使用新的模型管理架构
    
    优势：
    1. 清晰的模型生命周期管理
    2. 统一的检查点处理
    3. 灵活的参数配置
    4. 更好的可读性和维护性
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 创建环境
        self.env = self._create_environment()
        
        # 获取观察和动作规格
        self.observation_spec = self.env.observation_spec
        self.action_spec = self.env.action_spec
        
        # 创建模型管理器
        self.model_manager = self._create_model()
        
        # 打印模型摘要
        self.model_manager.print_model_summary()
        
        print("🎉 NavRL Trainer initialized successfully!")
    
    def _create_environment(self) -> ParallelEnv:
        """创建训练环境"""
        print("🏗️  Creating training environment...")
        
        # 创建单个环境实例
        env_fn = lambda: NavigationEnv(self.cfg)
        env = ParallelEnv(
            num_workers=self.cfg.env.num_envs,
            create_env_fn=env_fn,
            device=self.device
        )
        
        print(f"   - Environment created with {self.cfg.env.num_envs} parallel workers")
        return env
    
    def _create_model(self):
        """创建或加载模型"""
        # 检查是否有预训练检查点需要加载
        checkpoint_path = getattr(self.cfg, 'resume_checkpoint', None)
        
        if checkpoint_path and checkpoint_path != "":
            print(f"🔄 Loading from checkpoint: {checkpoint_path}")
            model_manager = load_pretrained_model(
                checkpoint_path=checkpoint_path,
                cfg=self.cfg,
                observation_spec=self.observation_spec,
                action_spec=self.action_spec,
                device=self.device,
                load_optimizer=True
            )
        else:
            print("🆕 Creating new model...")
            model_manager = create_navrl_model(
                cfg=self.cfg,
                observation_spec=self.observation_spec,
                action_spec=self.action_spec,
                device=self.device
            )
        
        return model_manager
    
    def train(self):
        """主训练循环"""
        print("🚀 Starting training...")
        
        model = self.model_manager.get_model()
        env = self.env
        
        # 训练循环
        for step in range(self.cfg.total_steps):
            # 1. 环境交互阶段
            with torch.no_grad():
                # 重置环境（如果需要）
                if step == 0 or step % self.cfg.rollout_length == 0:
                    td = env.reset()
                    print(f"   🔄 Environment reset at step {step}")
                
                # 收集轨迹数据
                rollout_data = self._collect_rollout(model, td, self.cfg.rollout_length)
            
            # 2. 模型训练阶段
            if len(rollout_data) >= self.cfg.batch_size:
                # 设置为训练模式
                self.model_manager.set_training_mode(True)
                
                # 执行训练更新
                train_info = self._train_step(model, rollout_data)
                
                # 记录训练信息
                if step % self.cfg.log_interval == 0:
                    self._log_training_info(step, train_info)
            
            # 3. 模型保存
            if step > 0 and step % self.cfg.save_interval == 0:
                self._save_checkpoint(step)
        
        print("✅ Training completed!")
    
    def _collect_rollout(self, model, td: TensorDict, rollout_length: int) -> list:
        """收集rollout数据"""
        rollout_data = []
        
        for _ in range(rollout_length):
            # 模型推理
            td = model(td)
            
            # 环境步进
            td = self.env.step(td)
            
            # 保存数据
            rollout_data.append(td.clone())
        
        return rollout_data
    
    def _train_step(self, model, rollout_data: list) -> dict:
        """执行一次训练步骤"""
        # 将rollout数据转换为训练批次
        batch_td = make_batch(rollout_data, self.device)
        
        # 模型训练
        train_info = model.train(batch_td)
        
        return train_info
    
    def _log_training_info(self, step: int, info: dict) -> None:
        """记录训练信息"""
        print(f"Step {step}:")
        for key, value in info.items():
            print(f"   {key}: {value:.4f}")
    
    def _save_checkpoint(self, step: int) -> None:
        """保存检查点"""
        save_path = f"{self.cfg.checkpoint_dir}/checkpoint_step_{step}.pt"
        
        self.model_manager.save_checkpoint(
            filepath=save_path,
            step=step,
            additional_info={'training_step': step}
        )
    
    def evaluate(self, num_episodes: int = 10) -> dict:
        """模型评估"""
        print(f"🔍 Evaluating model for {num_episodes} episodes...")
        
        # 设置为评估模式
        self.model_manager.set_training_mode(False)
        
        model = self.model_manager.get_model()
        total_rewards = []
        
        for episode in range(num_episodes):
            td = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    td = model(td)
                    td = self.env.step(td)
                    
                    episode_reward += td["next", "agents", "reward"].sum().item()
                    done = td["next", "terminated"].any().item()
            
            total_rewards.append(episode_reward)
        
        # 计算评估指标
        eval_metrics = {
            'mean_reward': sum(total_rewards) / len(total_rewards),
            'std_reward': torch.tensor(total_rewards).std().item(),
            'min_reward': min(total_rewards),
            'max_reward': max(total_rewards)
        }
        
        print(f"   📊 Evaluation results: {eval_metrics}")
        return eval_metrics


# === 高级使用示例 ===

def advanced_training_example(cfg):
    """
    高级训练示例：展示模型管理器的高级功能
    """
    print("🎯 Advanced Training Example")
    
    # 1. 创建训练器
    trainer = NavRLTrainer(cfg)
    
    # 2. 自定义ViT参数管理
    if cfg.training.freeze_vit_encoder:
        print("❄️  Freezing ViT encoder for stable fine-tuning...")
        trainer.model_manager.freeze_vit_encoder()
    
    # 3. 模型信息检查
    model_info = trainer.model_manager.get_model().get_model_info()
    print(f"📊 Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # 4. 训练
    trainer.train()
    
    # 5. 评估
    eval_results = trainer.evaluate()
    
    return trainer, eval_results


def fine_tuning_example(cfg, pretrained_checkpoint):
    """
    Fine-tuning示例：从预训练模型开始训练
    """
    print("🔧 Fine-tuning Example")
    
    # 加载预训练模型
    cfg.resume_checkpoint = pretrained_checkpoint
    trainer = NavRLTrainer(cfg)
    
    # 解冻ViT进行完整fine-tuning
    if cfg.training.full_vit_finetune:
        print("🔓 Unfreezing all ViT parameters for full fine-tuning...")
        trainer.model_manager.unfreeze_all_vit()
    
    # 训练和评估
    trainer.train()
    results = trainer.evaluate()
    
    return trainer, results


# === 配置示例 ===

@hydra.main(version_base=None, config_path="../cfg", config_name="train_ppo_vit")
def main(cfg: DictConfig):
    """
    主函数：使用新的模型管理器进行训练
    
    使用方法：
    python train_with_manager.py env.num_envs=64 training.freeze_vit_encoder=True
    """
    print("🎉 NavRL Training with Model Manager")
    print("="*50)
    
    # 基础训练
    if cfg.training.mode == "basic":
        trainer = NavRLTrainer(cfg)
        trainer.train()
        trainer.evaluate()
    
    # 高级训练
    elif cfg.training.mode == "advanced":
        trainer, results = advanced_training_example(cfg)
    
    # Fine-tuning
    elif cfg.training.mode == "finetune":
        trainer, results = fine_tuning_example(cfg, cfg.resume_checkpoint)
    
    else:
        print(f"❌ Unknown training mode: {cfg.training.mode}")
    
    print("🎉 Training session completed!")


if __name__ == "__main__":
    main()


# === 快速使用指南 ===
"""
快速使用指南：

1. 基础训练：
   ```python
   from models import create_navrl_model
   
   model_manager = create_navrl_model(cfg, obs_spec, act_spec, device)
   model = model_manager.get_model()
   # 使用model进行训练...
   ```

2. 加载预训练模型：
   ```python
   from models import load_pretrained_model
   
   model_manager = load_pretrained_model(checkpoint_path, cfg, obs_spec, act_spec, device)
   model = model_manager.get_model()
   ```

3. 参数管理：
   ```python
   # 冻结ViT encoder
   model_manager.freeze_vit_encoder()
   
   # 解冻所有ViT参数
   model_manager.unfreeze_all_vit()
   
   # 查看模型信息
   model_manager.print_model_summary()
   ```

4. 检查点管理：
   ```python
   # 保存检查点
   model_manager.save_checkpoint("checkpoint.pt", step=1000)
   
   # 加载检查点
   success = model_manager.load_checkpoint("checkpoint.pt")
   ```

优势：
- ✅ 代码更清晰，模块化程度高
- ✅ 统一的模型管理接口
- ✅ 灵活的参数配置和生命周期管理
- ✅ 更好的可读性和维护性
- ✅ 支持复杂的训练场景（预训练、fine-tuning等）
"""