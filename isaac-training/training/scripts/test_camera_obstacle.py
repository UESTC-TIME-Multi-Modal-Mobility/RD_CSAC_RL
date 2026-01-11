'''
Author: zdytim zdytim@foxmail.com
Date: 2026-01-02 22:16:16
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-02 22:16:17
FilePath: /NavRL/isaac-training/training/scripts/test_camera_obstacle.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
测试深度相机是否感知障碍物（符合 NavRL Copilot Guide）
"""

import torch
from omegaconf import OmegaConf
from env import NavigationEnv

# 加载配置
cfg = OmegaConf.load("training/cfg/train.yaml")
cfg.headless = False  # 可视化模式
cfg.env.num_envs = 1  # 单环境测试
cfg.env_dyn.num_obstacles = 4  # 启用动态障碍物

# 创建环境
env = NavigationEnv(cfg)
env.reset()

print(f"\n{'='*60}")
print(f"Camera Obstacle Detection Test")
print(f"{'='*60}")

# 运行 100 steps
for step in range(100):
    # 随机动作
    action = torch.randn(1, 1, 4, device=cfg.device) * 0.1
    obs = env.step(action)
    
    # 提取深度图
    depth = obs[("agents", "observation", "camera")][0, 0]  # [60, 90]
    
    # 分析深度图
    min_depth = depth.min().item()
    max_depth = depth.max().item()
    mean_depth = depth.mean().item()
    
    # 检测障碍物（深度 < 阈值）
    obstacle_mask = depth < 0.5  # 归一化深度 < 0.5（即 < 15m）
    obstacle_ratio = obstacle_mask.float().mean().item()
    
    if step % 10 == 0:
        print(f"\n[Step {step}]")
        print(f"  Depth range: [{min_depth:.4f}, {max_depth:.4f}]")
        print(f"  Mean depth: {mean_depth:.4f}")
        print(f"  Obstacle ratio: {obstacle_ratio*100:.1f}%")
        
        # 与 LiDAR 对比
        lidar = obs[("agents", "observation", "lidar")][0, 0]
        lidar_min = (cfg.sensor.lidar_range - lidar.max()).item()
        print(f"  LiDAR closest: {lidar_min:.2f}m")
        
        # ✅ 检查一致性
        if obstacle_ratio > 0.1 and lidar_min < 5.0:
            print(f"  ✅ Camera and LiDAR both detect obstacles")
        elif obstacle_ratio < 0.05 and lidar_min > 10.0:
            print(f"  ✅ Camera and LiDAR both clear")
        else:
            print(f"  ⚠️  Camera-LiDAR mismatch!")

print(f"\n{'='*60}")
print(f"Test complete. Check /tmp/navrl_depth_debug.png for visualization.")
print(f"{'='*60}\n")