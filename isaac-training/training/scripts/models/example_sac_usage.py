"""
SAC 模型使用示例
================

演示如何使用新的 SACModelManager 进行训练和推理

作者: NavRL Team
日期: 2026年1月6日
"""

import torch
from models.sac_model import SACModelManager, create_sac_model
from tensordict import TensorDict

# ============================================================================
# 示例 1: 基本使用
# ============================================================================

def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例 1: 基本使用")
    print("=" * 60)
    
    # 模拟配置
    class MockConfig:
        gamma = 0.99
        num_minibatches = 1
        
        class actor:
            learning_rate = 3e-4
            action_limit = 2.0
        
        class critic:
            learning_rate = 3e-4
        
        alpha_learning_rate = 3e-4
    
    cfg = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型管理器
    manager = SACModelManager(
        cfg=cfg,
        observation_spec=None,  # 将自动处理
        action_spec=2,  # 2维动作空间
        device=device
    )
    
    # 获取模型信息
    info = manager.get_model_info()
    print("\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ 基本使用示例完成\n")


# ============================================================================
# 示例 2: 推理（环境交互）
# ============================================================================

def example_inference():
    """推理示例"""
    print("=" * 60)
    print("示例 2: 推理（环境交互）")
    print("=" * 60)
    
    # 模拟配置（同上）
    class MockConfig:
        gamma = 0.99
        num_minibatches = 1
        class actor:
            learning_rate = 3e-4
            action_limit = 2.0
        class critic:
            learning_rate = 3e-4
        alpha_learning_rate = 3e-4
    
    cfg = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    manager = create_sac_model(cfg, None, 2, device)
    
    # 模拟观测
    batch_size = 4
    observations = {
        "lidar": torch.randn(batch_size, 1, 60, 60).to(device),
        "dynamic_obstacle": torch.randn(batch_size, 1, 10, 5).to(device),
        "state": torch.randn(batch_size, 8).to(device),
    }
    
    # 确定性推理（用于评估）
    with torch.no_grad():
        actions = manager.get_action(observations, deterministic=True)
    
    print(f"\n输入观测形状:")
    print(f"  lidar: {observations['lidar'].shape}")
    print(f"  dynamic_obstacle: {observations['dynamic_obstacle'].shape}")
    print(f"  state: {observations['state'].shape}")
    print(f"\n输出动作形状: {actions.shape}")
    print(f"动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    
    # 随机策略（用于探索）
    actions_stochastic = manager.get_action(observations, deterministic=False)
    print(f"\n随机动作形状: {actions_stochastic.shape}")
    
    print("\n✅ 推理示例完成\n")


# ============================================================================
# 示例 3: 训练步骤
# ============================================================================

def example_training():
    """训练示例"""
    print("=" * 60)
    print("示例 3: 训练步骤")
    print("=" * 60)
    
    # 配置
    class MockConfig:
        gamma = 0.99
        num_minibatches = 2
        class actor:
            learning_rate = 3e-4
            action_limit = 2.0
        class critic:
            learning_rate = 3e-4
        alpha_learning_rate = 3e-4
    
    cfg = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    manager = create_sac_model(cfg, None, 2, device)
    
    # 模拟经验回放缓冲区（这里需要真实的 ReplayBuffer）
    # 注：实际使用时需要从 TorchRL 导入 ReplayBuffer
    print("\n⚠️  训练需要真实的 ReplayBuffer，这里仅演示接口")
    print("训练调用方式:")
    print("  loss_info = manager.train_step(replay_buffer, batch_size=256, tau=0.005)")
    print("\n返回的损失信息包括:")
    print("  - actor_loss: Actor 损失")
    print("  - q1_loss, q2_loss: Critic 损失")
    print("  - alpha_loss: Temperature 损失")
    print("  - alpha: 当前 temperature 值")
    print("  - 其他训练指标...")
    
    print("\n✅ 训练示例说明完成\n")


# ============================================================================
# 示例 4: 检查点保存和加载
# ============================================================================

def example_checkpoint():
    """检查点管理示例"""
    print("=" * 60)
    print("示例 4: 检查点保存和加载")
    print("=" * 60)
    
    # 配置
    class MockConfig:
        gamma = 0.99
        num_minibatches = 1
        class actor:
            learning_rate = 3e-4
            action_limit = 2.0
        class critic:
            learning_rate = 3e-4
        alpha_learning_rate = 3e-4
    
    cfg = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    manager = create_sac_model(cfg, None, 2, device)
    
    # 保存检查点
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
        
        print(f"\n保存检查点到: {checkpoint_path}")
        manager.save_checkpoint(checkpoint_path, step=1000, extra_info="test")
        
        # 修改模型参数（模拟训练）
        original_alpha = manager.model.alpha.item()
        manager.model.log_alpha.data += 0.5
        manager.model.alpha = manager.model.log_alpha.exp().detach()
        modified_alpha = manager.model.alpha.item()
        
        print(f"\n修改前 alpha: {original_alpha:.4f}")
        print(f"修改后 alpha: {modified_alpha:.4f}")
        
        # 加载检查点
        print(f"\n从检查点加载...")
        checkpoint = manager.load_checkpoint(checkpoint_path)
        restored_alpha = manager.model.alpha.item()
        
        print(f"恢复后 alpha: {restored_alpha:.4f}")
        print(f"检查点步数: {checkpoint['step']}")
        
        assert abs(restored_alpha - original_alpha) < 1e-6, "参数恢复失败"
        print("\n✅ 检查点验证通过")
    
    print("\n✅ 检查点示例完成\n")


# ============================================================================
# 示例 5: 向后兼容（使用 SAC_V2）
# ============================================================================

def example_backward_compatibility():
    """向后兼容示例"""
    print("=" * 60)
    print("示例 5: 向后兼容（SAC_V2）")
    print("=" * 60)
    
    # 配置
    class MockConfig:
        gamma = 0.99
        num_minibatches = 1
        class actor:
            learning_rate = 3e-4
            action_limit = 2.0
        class critic:
            learning_rate = 3e-4
        alpha_learning_rate = 3e-4
    
    cfg = MockConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用 SAC_V2（向后兼容接口）
    from SAC_v1 import SAC_V2
    
    agent = SAC_V2(cfg, None, 2, device)
    
    print("\n✅ SAC_V2 创建成功")
    print("可以像使用原始 SAC 一样使用 SAC_V2：")
    print("  - agent.get_action(state)")
    print("  - agent(tensordict)")
    print("  - agent.train(replay_buffer, batch_size)")
    print("  - agent.save_checkpoint(...)")
    print("  - agent.load_checkpoint(...)")
    
    print("\n✅ 向后兼容示例完成\n")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SAC 模型管理器使用示例")
    print("=" * 60 + "\n")
    
    # 运行所有示例
    example_basic_usage()
    example_inference()
    example_training()
    example_checkpoint()
    example_backward_compatibility()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60 + "\n")
