'''
Author: zdytim zdytim@foxmail.com
Date: 2025-08-19 15:40:20
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2025-08-19 15:55:50
FilePath: /u20/NavRL/isaac-training/training/scripts/train_sac_corrected.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# -*- coding: utf-8 -*-
import torch
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict.tensordict import TensorDict
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 优化的配置
class SAC_Config:
    class Actor:
        learning_rate = 1e-4   
        action_limit = 2.0
    class Critic:
        learning_rate = 1e-3  
    actor = Actor()
    critic = Critic()
    gamma = 0.99
    alpha_learning_rate = 5e-4
    target_entropy = -1.0

cfg = SAC_Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 环境与Agent
env = GymEnv("Pendulum-v1", device=device)
env = TransformedEnv(env)
observation_spec = env.observation_spec
action_spec = env.action_spec
print("Observation spec:", observation_spec)
print("Action spec:", action_spec)

from SAC_improved import SAC
agent = SAC(cfg, observation_spec, action_spec, device)

# Replay buffer
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(10000),
    batch_size=200,
)

def policy(td):
    obs = td['observation']
    action = agent.get_action(obs, deterministic=False)
    return TensorDict({'action': action}, batch_size=td.batch_size)

collector = SyncDataCollector(
    env,
    policy=policy,
    frames_per_batch=200,
    total_frames=1_000_0000,
    device=device,
    storing_device=device,
    reset_at_each_iter=True,
    max_frames_per_traj=200,
)


# 统计变量
episode_rewards = []
episode_lengths = []
metrics = {
    'actor_losses': [],
    'critic_losses': [],
    'alpha_values': [],
    'iterations': [],
    'frames': [],
    'smoothed_rewards': [],
}
cur_frames = 0
ep_reward = 0
ep_length = 0

print("Starting improved SAC training with proper episode tracking...")

for n_iter, batch in enumerate(collector):
    # 统计帧数增量（更稳健）
    batch_rewards = batch["next", "reward"].view(-1)
    batch_dones = (batch["next", "terminated"] | batch["next", "truncated"]).view(-1)
    n_batch_frames = batch_rewards.numel()
    cur_frames += n_batch_frames

    # 跨batch统计episode
    for r, done in zip(batch_rewards, batch_dones):
        ep_reward += r.item()
        ep_length += 1
        if done.item():
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            ep_reward = 0
            ep_length = 0

    # 存到replay buffer
    for transition in batch:
        replay_buffer.add(transition)

    # 训练
    loss_info = {}
    if len(replay_buffer) > 2000:
        loss_info, _ = agent.train(replay_buffer, batch_size=200, tau=0.005)
        metrics['actor_losses'].append(loss_info.get('actor_loss', 0))
        metrics['critic_losses'].append(loss_info.get('critic1_loss', 0))
        metrics['alpha_values'].append(loss_info.get('alpha', 0))
        metrics['iterations'].append(n_iter)
        metrics['frames'].append(cur_frames)

    # 平滑奖励
    if len(episode_rewards) >= 10:
        smoothed_reward = np.mean(episode_rewards[-10:])
        metrics['smoothed_rewards'].append(smoothed_reward)

    if n_iter % 50 == 0:
        last_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
        last_lengths = episode_lengths[-10:] if len(episode_lengths) >= 10 else episode_lengths
        recent_reward = np.mean(last_rewards) if last_rewards else 0
        recent_length = np.mean(last_lengths) if last_lengths else 0
        print(f'Iter {n_iter}, frames={cur_frames}, episodes={len(episode_rewards)}, avg_reward={recent_reward:.2f}, avg_length={recent_length:.1f}')
        if len(replay_buffer) > 2000:
            print("  Losses - Actor: {:.4f}, Critic: {:.4f}, Alpha: {:.4f}".format(
                loss_info.get('actor_loss', 0), 
                loss_info.get('critic1_loss', 0),
                loss_info.get('alpha', 0)
            ))

    # 按帧数终止更直观
    if cur_frames > 3_000_000 or loss_info.get('alpha', 0) < 0.001:
        break

print(f"Training finished! Total episodes: {len(episode_rewards)}")

# 可视化
plots_dir = "training_plots"
os.makedirs(plots_dir, exist_ok=True)

plt.rcParams.update({'font.size': 11})
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('SAC Training Results - Pendulum-v1 (Corrected Episode Rewards)', 
             fontsize=16, fontweight='bold')

# Plot 1: Episode Rewards
episodes_x = list(range(1, len(episode_rewards) + 1))
axes[0, 0].plot(episodes_x, episode_rewards, alpha=0.4, color='lightblue', linewidth=0.8)
if len(episode_rewards) >= 20:
    window = 20
    moving_avg = [np.mean(episode_rewards[max(0, i-window):i+1]) for i in range(len(episode_rewards))]
    axes[0, 0].plot(episodes_x, moving_avg, color='darkblue', linewidth=2.5, label=f'Moving Average ({window})')
    axes[0, 0].legend()
axes[0, 0].set_title('Episode Rewards Over Time')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Episode Reward')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Episode Lengths
axes[0, 1].plot(episodes_x, episode_lengths, alpha=0.6, color='green', linewidth=1)
axes[0, 1].set_title('Episode Lengths')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Steps per Episode')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Actor Loss
if metrics['actor_losses']:
    axes[0, 2].plot(metrics['iterations'], metrics['actor_losses'], color='red', linewidth=1.5)
    axes[0, 2].set_title('Actor Loss')
    axes[0, 2].set_xlabel('Training Iteration')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Critic Loss
if metrics['critic_losses']:
    axes[1, 0].plot(metrics['iterations'], metrics['critic_losses'], color='orange', linewidth=1.5)
    axes[1, 0].set_title('Critic Loss')
    axes[1, 0].set_xlabel('Training Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Alpha Values
if metrics['alpha_values']:
    axes[1, 1].plot(metrics['iterations'], metrics['alpha_values'], color='purple', linewidth=1.5)
    axes[1, 1].set_title('Temperature Parameter (Alpha)')
    axes[1, 1].set_xlabel('Training Iteration')
    axes[1, 1].set_ylabel('Alpha')
    axes[1, 1].grid(True, alpha=0.3)

# Plot 6: 学习进度（最近100个episode的平均奖励）
if len(episode_rewards) >= 100:
    progress_data = []
    for i in range(100, len(episode_rewards), 20):
        avg_reward = np.mean(episode_rewards[i-100:i])
        progress_data.append(avg_reward)
    x_progress = list(range(100, len(episode_rewards), 20))[:len(progress_data)]
    if progress_data:
        axes[1, 2].plot(x_progress, progress_data, 'o-', color='brown', linewidth=2, markersize=4)
        axes[1, 2].set_title('Learning Progress (100-ep Average)')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Average Reward')
        axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'{plots_dir}/sac_corrected_training_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

# 测试训练好的策略
print("Testing trained policy...")
import gymnasium as gym

test_rewards = []
test_lengths = []
num_test_episodes = 15

for episode in range(num_test_episodes):
    test_env = gym.make("Pendulum-v1")
    obs, info = test_env.reset()
    episode_reward = 0
    steps = 0
    for step in range(200):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            action = agent.get_action(obs_tensor, deterministic=True)
            # 保证action shape为(action_dim,)
            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu().numpy().squeeze()
            else:
                action_np = np.asarray(action).squeeze()
            action_np = np.array(action_np).reshape(-1) 
        obs, reward, terminated, truncated, info = test_env.step(action_np)
        episode_reward += reward
        steps += 1
        if terminated or truncated:
            break
    test_rewards.append(episode_reward)
    test_lengths.append(steps)
    if episode < 3:
        print(f"Test Episode {episode + 1}: {episode_reward:.2f} (steps: {steps})")

mean_test_reward = np.mean(test_rewards)
std_test_reward = np.std(test_rewards)
mean_test_length = np.mean(test_lengths)

# 安全切片，防止episode数量太少报错
last_rewards = episode_rewards[-20:] if len(episode_rewards) >= 20 else episode_rewards
last_50 = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
initial_avg = np.mean(episode_rewards[:50]) if len(episode_rewards) >= 50 else (np.mean(episode_rewards[:10]) if episode_rewards else 0)
final_avg = np.mean(last_50) if last_50 else 0

print(f"\n📊 Final Results Summary:")
print(f"Training Episodes: {len(episode_rewards)}")
print(f"Final Training Average (last 20): {np.mean(last_rewards) if last_rewards else 0:.2f}")
print(f"Test Performance: {mean_test_reward:.2f} ± {std_test_reward:.2f}")
print(f"Average Test Length: {mean_test_length:.1f} steps")
print(f"Best Test: {max(test_rewards) if test_rewards else 0:.2f}")
print(f"Worst Test: {min(test_rewards) if test_rewards else 0:.2f}")

print(f"\n🎯 Training Analysis:")
print(f"Initial Average: {initial_avg:.2f}")
print(f"Final Average: {final_avg:.2f}")
print(f"Improvement: {final_avg - initial_avg:.2f}")

if mean_test_reward > -500:
    status = "✅ Excellent performance!"
elif mean_test_reward > -800:
    status = "🟡 Good performance"
elif mean_test_reward > -1200:
    status = "🟠 Moderate performance"
else:
    status = "❌ Needs improvement"
print(f"Performance Status: {status}")

# 测试结果可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Policy Evaluation - Test Results', fontsize=14, fontweight='bold')

colors = ['darkred' if r < -1200 else 'orange' if r < -800 else 'green' for r in test_rewards]
ax1.bar(range(1, len(test_rewards)+1), test_rewards, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=mean_test_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_test_reward:.1f}')
ax1.set_title('Test Episode Performance')
ax1.set_xlabel('Test Episode')
ax1.set_ylabel('Total Reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.hist(test_rewards, bins=6, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(x=mean_test_reward, color='red', linestyle='--', linewidth=2)
ax2.set_title('Reward Distribution')
ax2.set_xlabel('Total Reward')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plots_dir}/test_evaluation_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n💾 All plots saved to '{plots_dir}/' directory")
print("🎉 SAC training and evaluation completed!")