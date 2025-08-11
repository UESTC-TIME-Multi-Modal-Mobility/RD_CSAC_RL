# -*- coding: utf-8 -*-
'''
Author: zdytim zdytim@foxmail.com
Date: 2025-08-05 12:30:25
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2025-08-11 15:55:05
FilePath: /u20/NavRL/isaac-training/training/scripts/DummyEnv_train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import OmegaConf
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.envs.utils import ExplorationType

from SAC import SAC
from utils import evaluate

# 可选：DummyEnv用于快速调试
from tensordict import TensorDict

class DummyEnv:
    def __init__(self, num_envs=50, num_frames=32, device="cuda:0"):
        self.num_envs = num_envs
        self.num_frames = num_frames
        self.device = device
        
        # 伪造 observation_spec 和 action_spec，适配SAC初始化
        class DummySpec:
            def __init__(self_inner, device=None):
                self_inner._device = torch.device(device) if device is not None else torch.device(self.device)
            
            def zero(self_inner):
                # 返回一个与真实环境观测结构完全一致的TensorDict
                return TensorDict({
                    "agents": TensorDict({
                        "observation": TensorDict({
                            "lidar": torch.zeros(1, 1, 36, 4, device=self_inner._device),
                            "dynamic_obstacle": torch.zeros(1, 1, 5, 10, device=self_inner._device),
                            "state": torch.zeros(1, 8, device=self_inner._device),
                            "direction": torch.zeros(1, 1, 3, device=self_inner._device),
                        }, batch_size=[1], device=self_inner._device)
                    }, batch_size=[1], device=self_inner._device)
                }, batch_size=[1], device=self_inner._device)
            
            @property
            def shape(self_inner):
                return (1, 3)  # 根据SAC输出: agents.action shape [50, 32, 1, 3]
            
            def clone(self_inner):
                # 添加clone方法，返回自身的副本
                return DummySpec(self_inner._device)
            
            @property
            def device(self_inner):
                return self_inner._device
            
            @property
            def dtype(self_inner):
                return torch.float32
            
            def to(self_inner, dest):
                # 添加to方法，支持设备转换
                return DummySpec(dest)
                
        self.observation_spec = DummySpec(self.device)
        self.action_spec = DummySpec(self.device)

    def generate_sample_data(self):
        """生成一个完整的采样数据，模拟SyncDataCollector的输出结构"""
        return TensorDict({
            # 原始观测数据
            "agents": TensorDict({
                "observation": TensorDict({
                    "lidar": torch.randn(self.num_envs, self.num_frames, 1, 36, 4, device=self.device),
                    "dynamic_obstacle": torch.randn(self.num_envs, self.num_frames, 1, 5, 10, device=self.device),
                    "state": torch.randn(self.num_envs, self.num_frames, 8, device=self.device),
                    "direction": torch.randn(self.num_envs, self.num_frames, 1, 3, device=self.device),
                }, batch_size=[self.num_envs, self.num_frames], device=self.device),
                "action_normalized": torch.randn(self.num_envs, self.num_frames, 1, 3, device=self.device),  # SAC输出的动作
            }, batch_size=[self.num_envs, self.num_frames], device=self.device),
            
            # 特征提取器输出（SAC网络会生成这些）
            "_cnn_feature": torch.randn(self.num_envs, self.num_frames, 128, device=self.device),
            "_dynamic_obstacle_feature": torch.randn(self.num_envs, self.num_frames, 64, device=self.device),
            "_feature": torch.randn(self.num_envs, self.num_frames, 256, device=self.device),
            "actor_params": torch.randn(self.num_envs, self.num_frames, 6, device=self.device),  # loc+scale参数
            "loc": torch.randn(self.num_envs, self.num_frames, 3, device=self.device),
            "scale": torch.abs(torch.randn(self.num_envs, self.num_frames, 3, device=self.device)) + 0.1,  # 保证正数
            "sample_log_prob": torch.randn(self.num_envs, self.num_frames, device=self.device),
            
            # 环境状态信息
            "done": torch.rand(self.num_envs, self.num_frames, 1, device=self.device) > 0.95,
            "terminated": torch.rand(self.num_envs, self.num_frames, 1, device=self.device) > 0.95,
            "truncated": torch.rand(self.num_envs, self.num_frames, 1, device=self.device) > 0.98,
            
            # 轨迹追踪
            "collector": TensorDict({
                "traj_ids": torch.randint(0, 1000, (self.num_envs, self.num_frames), device=self.device, dtype=torch.int64),
            }, batch_size=[self.num_envs, self.num_frames], device=self.device),
            
            # 环境信息
            "info": TensorDict({
                "drone_state": torch.randn(self.num_envs, self.num_frames, 1, 13, device=self.device)
            }, batch_size=[self.num_envs, self.num_frames], device=self.device),
            
            # 统计信息
            "stats": TensorDict({
                "collision": torch.randint(0, 2, (self.num_envs, self.num_frames, 1), device=self.device, dtype=torch.float32),
                "episode_len": torch.randint(1, 1000, (self.num_envs, self.num_frames, 1), device=self.device, dtype=torch.float32),
                "reach_goal": torch.randint(0, 2, (self.num_envs, self.num_frames, 1), device=self.device, dtype=torch.float32),
                "return": torch.randn(self.num_envs, self.num_frames, 1, device=self.device),
                "truncated": torch.zeros(self.num_envs, self.num_frames, 1, device=self.device, dtype=torch.float32),
            }, batch_size=[self.num_envs, self.num_frames], device=self.device),
            
            # next状态（强化学习训练必须）
            "next": TensorDict({
                "agents": TensorDict({
                    "observation": TensorDict({
                        "lidar": torch.randn(self.num_envs, self.num_frames, 1, 36, 4, device=self.device),
                        "dynamic_obstacle": torch.randn(self.num_envs, self.num_frames, 1, 5, 10, device=self.device),
                        "state": torch.randn(self.num_envs, self.num_frames, 8, device=self.device),
                        "direction": torch.randn(self.num_envs, self.num_frames, 1, 3, device=self.device),
                    }, batch_size=[self.num_envs, self.num_frames], device=self.device),
                    "reward": torch.randn(self.num_envs, self.num_frames, 1, device=self.device),
                }, batch_size=[self.num_envs, self.num_frames], device=self.device),
                "done": torch.rand(self.num_envs, self.num_frames, 1, device=self.device) > 0.95,
                "terminated": torch.rand(self.num_envs, self.num_frames, 1, device=self.device) > 0.95,
                "truncated": torch.rand(self.num_envs, self.num_frames, 1, device=self.device) > 0.98,
                "info": TensorDict({
                    "drone_state": torch.randn(self.num_envs, self.num_frames, 1, 13, device=self.device)
                }, batch_size=[self.num_envs, self.num_frames], device=self.device),
                "stats": TensorDict({
                    "collision": torch.randint(0, 2, (self.num_envs, self.num_frames, 1), device=self.device, dtype=torch.float32),
                    "episode_len": torch.randint(1, 1000, (self.num_envs, self.num_frames, 1), device=self.device, dtype=torch.float32),
                    "reach_goal": torch.randint(0, 2, (self.num_envs, self.num_frames, 1), device=self.device, dtype=torch.float32),
                    "return": torch.randn(self.num_envs, self.num_frames, 1, device=self.device),
                    "truncated": torch.zeros(self.num_envs, self.num_frames, 1, device=self.device, dtype=torch.float32),
                }, batch_size=[self.num_envs, self.num_frames], device=self.device),
            }, batch_size=[self.num_envs, self.num_frames], device=self.device),
            
        }, batch_size=[self.num_envs, self.num_frames], device=self.device)

    def reset(self):
        # 简化版本，只返回观测部分
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "lidar": torch.randn(self.num_envs, self.num_frames, 1, 36, 4, device=self.device),
                    "dynamic_obstacle": torch.randn(self.num_envs, self.num_frames, 1, 5, 10, device=self.device),
                    "state": torch.randn(self.num_envs, self.num_frames, 8, device=self.device),
                    "direction": torch.randn(self.num_envs, self.num_frames, 1, 3, device=self.device),
                }, batch_size=[self.num_envs, self.num_frames], device=self.device)
            }, batch_size=[self.num_envs, self.num_frames], device=self.device)
        }, batch_size=[self.num_envs, self.num_frames], device=self.device)
        return obs

    def step(self, action):
        return self.generate_sample_data()

@hydra.main(config_path="../cfg", config_name="train", version_base=None)
def main(cfg):
    # 代理环境选择
    use_dummy_env = getattr(cfg, "use_dummy_env", True)
    device = cfg.device if hasattr(cfg, "device") else "cuda:0"

    # wandb初始化
    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        id=wandb.util.generate_id(),
    )

    # 环境初始化
    if use_dummy_env:
        env = DummyEnv(num_envs=cfg.env.num_envs, num_frames=cfg.algo.training_frame_num, device=device)
        transformed_env = env  # DummyEnv不需要transforms
    else:
        from omni.isaac.kit import SimulationApp
        sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
        from env import NavigationEnv
        from omni_drones.controllers import LeePositionController
        from omni_drones.utils.torchrl.transforms import VelController
        from torchrl.envs.transforms import TransformedEnv, Compose

        env = NavigationEnv(cfg)
        controller = LeePositionController(9.81, env.drone.params).to(device)
        vel_transform = VelController(controller, yaw_control=False)
        transformed_env = TransformedEnv(env, Compose(vel_transform)).train()
        transformed_env.set_seed(cfg.seed)

    # SAC Policy
    policy = SAC(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, device)

    # Replay Buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.algo.buffer_size, device="cpu"),
        batch_size=cfg.algo.batch_size if hasattr(cfg.algo, 'batch_size') else 128,
    )

    # Collector
    if use_dummy_env:
        # DummyCollector：模拟SyncDataCollector的行为
        class DummyCollector:
            def __init__(self, env):
                self.env = env
                self._frames = 0
                self._fps = 60.0  # 模拟FPS
                
            def __iter__(self):
                return self
                
            def __next__(self):
                # 每次调用返回一个完整的采样数据
                self._frames += self.env.num_envs * self.env.num_frames
                return self.env.generate_sample_data()
        
        collector = DummyCollector(env)
    else:
        from omni_drones.utils.torchrl import SyncDataCollector
        from torchrl.envs.utils import ExplorationType
        collector = SyncDataCollector(
            transformed_env,
            policy=policy,
            frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,
            total_frames=cfg.max_frame_num,
            device=device,
            return_same_td=True,
            exploration_type=ExplorationType.RANDOM,
        )

    # 训练主循环
    update_counter = 0
    warmup_steps = cfg.algo.warmup_steps if hasattr(cfg.algo, 'warmup_steps') else 1000
    eval_interval = cfg.eval_interval if hasattr(cfg, "eval_interval") else 1000
    save_interval = cfg.save_interval if hasattr(cfg, "save_interval") else 1000

    try:
        for i, data in enumerate(collector):
            replay_buffer.extend(data)
            info = {"env_frames": i, "rollout_fps": 0}

            if replay_buffer.__len__() >= warmup_steps:
                train_loss_stats,rewards = policy.train(replay_buffer)
                info.update(train_loss_stats)
                info.update(rewards)
                update_counter += 1
            else:
                info.update({"status": "warming_up", "buffer_size": replay_buffer.__len__()})

            # 评估
            if i % eval_interval == 0:
                if not use_dummy_env:
                    env.enable_render(True)
                    env.eval()
                    eval_info = evaluate(
                        env=transformed_env,
                        policy=policy,
                        seed=cfg.seed,
                        cfg=cfg,
                        exploration_type=ExplorationType.MEAN
                    )
                    env.enable_render(not cfg.headless)
                    env.train()
                    env.reset()
                    info.update(eval_info)
                    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [NavRL]: evaluation done.")
                else:
                    # DummyEnv模式下的简化评估
                    eval_info = {
                        "eval/episode_reward": torch.randn(1).item() * 100,
                        "eval/success_rate": torch.rand(1).item(),
                        "eval/collision_rate": torch.rand(1).item() * 0.1,
                        "eval/episode_length": torch.randint(50, 500, (1,)).item(),
                    }
                    info.update(eval_info)
                    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DummyEnv]: mock evaluation done.")

            # wandb日志
            run.log(info)
            print(f"\n[Step {i}] SAC Training Summary:")
            for k, v in info.items():
                if isinstance(v, (float, int)):
                    print(f"  {k:<20}: {v:.4f}")
            print(f"  Buffer Size: {replay_buffer.__len__()}")
            print(f"  Updates: {update_counter}")
            print("-" * 40)

            # 保存模型
            if i % save_interval == 0:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                print("[NavRL]: model saved at training step: ", i)

        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)
        wandb.finish()
        if not use_dummy_env:
            sim_app.close()
    except KeyboardInterrupt:
        print("\n[NavRL]: KeyboardInterrupt received. Saving model and finishing wandb run.")
        ckpt_path = os.path.join(run.dir, f"checkpoint_interrupt.pt")
        torch.save(policy.state_dict(), ckpt_path)
        run.log({"interrupted_at_step": i})
        run.finish()
        if not use_dummy_env:
            sim_app.close()
        exit(0)

if __name__ == "__main__":
    main()