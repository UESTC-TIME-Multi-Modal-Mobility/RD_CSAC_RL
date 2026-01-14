'''
Author: zdytim zdytim@foxmail.com
Date: 2026-01-14 23:20:20
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-15 00:05:06
FilePath: /NavRL/isaac-training/training/scripts/RevKDTrainRL.py
Description: RL-based Knowledge Distillation using Teacher's Critic to guide Student Actor
'''
import argparse
import os
import hydra
import datetime
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from SAC_lag import SAC
from models.navrl_model import NavRLModel
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from utils import evaluate, vec_to_world
from torchrl.envs.utils import ExplorationType
import datetime

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

FILE_PATH = os.path.join(os.path.dirname(__file__), "../cfg")


def compute_rl_kd_loss(student_actions, obs, teacher_critic1, teacher_critic2, 
                        teacher_actions=None, kd_alpha=0.8, bc_weight=0.2):
    """
    RL-based Knowledge Distillation损失计算
    
    使用Teacher的Critic网络评估Student的动作质量，引导Student最大化Q值
    同时可选地加入行为克隆损失作为辅助
    
    Args:
        student_actions: Student policy输出的动作 [batch_size, action_dim]
        obs: 观测数据字典，包含 lidar, dynamic_obstacle, state
        teacher_critic1, teacher_critic2: Teacher的Critic网络 (frozen)
        teacher_actions: Teacher的动作 (用于可选的BC损失)
        kd_alpha: RL损失权重
        bc_weight: 行为克隆损失权重
    
    Returns:
        total_loss: 总损失
        loss_dict: 详细损失字典
    """
    # CriticNetwork.forward(s, a) 期望:
    # s: dict with keys "lidar", "dynamic_obstacle", "state"
    # a: action tensor [batch, action_dim]
    # 它内部会创建 TensorDict({"observation": s, "action_normalized": a})
    
    # obs 应该已经是包含这些键的字典或TensorDict
    # 需要确保有 "lidar" 键（SAC Critic使用lidar，不是camera）
    if "lidar" not in obs.keys():
        raise ValueError("Teacher Critic requires 'lidar' in observation. "
                        "Make sure the environment provides both camera and lidar observations.")
    
    # 使用Teacher Critic评估Student动作的Q值
    # 梯度会通过student_actions流回Student网络
    q1_student = teacher_critic1(obs, student_actions).squeeze(-1)
    q2_student = teacher_critic2(obs, student_actions).squeeze(-1)
    q_min_student = torch.min(q1_student, q2_student)
    
    # RL损失：最大化Teacher Critic对Student动作的Q值评估
    # 最大化Q值 = 最小化 -Q值
    rl_loss = -q_min_student.mean()
    
    # 可选的行为克隆损失（辅助Student学习Teacher的动作分布）
    bc_loss = torch.tensor(0.0, device=student_actions.device)
    if teacher_actions is not None and bc_weight > 0:
        bc_loss = F.mse_loss(student_actions, teacher_actions.detach())
    
    # 总损失组合
    total_loss = kd_alpha * rl_loss + bc_weight * bc_loss
    
    # 详细统计（使用detach避免计算图问题）
    with torch.no_grad():
        loss_dict = {
            "kd_rl_loss": rl_loss.item(),
            "kd_bc_loss": bc_loss.item() if bc_weight > 0 else 0.0,
            "kd_total_loss": total_loss.item(),
            "q_student_mean": q_min_student.mean().item(),
            "q_student_std": q_min_student.std().item(),
            "q_student_min": q_min_student.min().item(),
            "q_student_max": q_min_student.max().item(),
        }
        
        # 如果有teacher_actions，计算动作差异统计
        if teacher_actions is not None:
            action_diff = (student_actions - teacher_actions).abs()
            loss_dict.update({
                "action_diff_vx": action_diff[:, 0].mean().item(),
                "action_diff_vy": action_diff[:, 1].mean().item(),
                "action_diff_vz": action_diff[:, 2].mean().item(),
                "action_diff_total": action_diff.mean().item(),
            })
    
    return total_loss, loss_dict
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Wandb initialization
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"KD_{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"KD_{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # Environment setup
    from env import NavigationEnv
    env = NavigationEnv(cfg)
    
    transforms = []
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)

    # Initialize teacher and student policies
    policy_t = SAC(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    policy_s = NavRLModel(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    
    # Load pretrained teacher model
    teacher_ckpt_path = cfg.get('teacher_checkpoint', "/home/u20/NavRL/isaac-training/training/scripts/checkpoint/CSAC_0_2100.pt")
    policy_t.load_state_dict(torch.load(teacher_ckpt_path, map_location=cfg.device))
    print(f"[NavRL KD]: Teacher loaded from {teacher_ckpt_path}")
    
    # Freeze teacher completely - both parameters and Critic networks
    for param in policy_t.parameters():
        param.requires_grad = False
    policy_t.training = False
    
    # 重要：确保Teacher的Critic网络也被冻结但保持可调用
    policy_t.critic1.eval()
    policy_t.critic2.eval()
    policy_t.critic1_cost1.eval()
    policy_t.critic2_cost2.eval()
    
    print(f"[NavRL KD]: Teacher frozen with Critic networks: critic1, critic2")
    student_ckpt_path = cfg.get('student_checkpoint', "/home/u20/NavRL/isaac-training/training/scripts/checkpoint/student_checkpoint_49000.pt")
    policy_s.load_state_dict(torch.load(student_ckpt_path, map_location=cfg.device))
    # Student remains in training mode
    policy_s.training = True
    for param in policy_s.parameters():
        param.requires_grad = True

    # Replay buffer for storing teacher-generated data
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.algo.buffer_size,
        ),
        batch_size=cfg.algo.batch_size,
    )

    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # Data collector using teacher policy for sampling
    collector = SyncDataCollector(
        transformed_env,
        policy=policy_t,
        frames_per_batch=cfg.env.num_envs * cfg.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True,
        exploration_type=ExplorationType.MEAN,
    )

    # RL-based Knowledge distillation parameters
    kd_alpha = cfg.get('kd_alpha', 0.8)       # RL损失权重
    bc_weight = cfg.get('bc_weight', 0.2)     # 行为克隆损失权重
    update_counter = 0
    warmup_steps = cfg.algo.warmup_steps
    
    training_epoch_num = cfg.get('training_epoch_num', 3)
    print(f"[NavRL RL-KD]: kd_alpha (RL weight): {kd_alpha}")
    print(f"[NavRL RL-KD]: bc_weight (BC weight): {bc_weight}")
    print(f"[NavRL RL-KD]: training_epoch_num: {training_epoch_num}")
    print(f"[NavRL RL-KD]: Using Teacher Critic to guide Student Actor!")

    try:
        for i, data in enumerate(collector):
            # Store teacher-generated data
            data = data.reshape(-1).cpu()
            replay_buffer.extend(data)
            
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            
            if len(replay_buffer) >= warmup_steps:
                batch_losses = []
                batch_lr_values = []
                
                for update_idx in range(training_epoch_num):
                    # Sample batch from teacher experience buffer
                    batch = replay_buffer.sample().to(cfg.device)
                    obs = batch[("agents", "observation")]
                    
                    # Teacher inference for BC reference (optional)
                    with torch.no_grad():
                        teacher_output = policy_t(batch)
                        teacher_actions = teacher_output[("agents", "action_normalized")]
                    
                    # Student forward pass
                    # 提取特征
                    student_latent = policy_s.shared_features(
                        obs["camera"],
                        obs["dynamic_obstacle"], 
                        obs["state"]
                    )
                    
                    # 创建临时tensordict for actor
                    temp_td = batch.clone()
                    temp_td.set("_latent", student_latent)
                    
                    # Get student actions (这里需要有梯度流)
                    policy_s.actor_head(temp_td)
                    student_actions = temp_td[("agents", "action_normalized")]
                    
                    # RL-based Knowledge Distillation: 使用Teacher Critic评估Student动作
                    # 核心思想：Student最大化Teacher Critic对其动作的Q值评估
                    kd_loss, kd_loss_dict = compute_rl_kd_loss(
                        student_actions=student_actions,
                        obs=obs,
                        teacher_critic1=policy_t.critic1,
                        teacher_critic2=policy_t.critic2,
                        teacher_actions=teacher_actions,
                        kd_alpha=kd_alpha,
                        bc_weight=bc_weight
                    )
                    
                    # Student parameter update
                    policy_s.optimizer.zero_grad()
                    kd_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_s.parameters(), max_norm=1.0)
                    policy_s.optimizer.step()
                    
                    # 收集统计信息
                    batch_losses.append(kd_loss.item())
                    batch_lr_values.append(policy_s.optimizer.param_groups[0]['lr'])
                    
                    # 累积损失统计用于平均
                    if update_idx == 0:
                        batch_kd_losses = {k: [v] for k, v in kd_loss_dict.items()}
                    else:
                        for k, v in kd_loss_dict.items():
                            batch_kd_losses[k].append(v)
                    
                    update_counter += 1
                
                # 计算平均损失和训练统计
                if batch_losses:
                    avg_kd_loss = sum(batch_losses) / len(batch_losses)
                    avg_lr = sum(batch_lr_values) / len(batch_lr_values)
                    
                    # 计算各项损失的平均值
                    avg_kd_losses = {
                        f"train/{k}": sum(v) / len(v) 
                        for k, v in batch_kd_losses.items()
                    }
                    
                    # Log training statistics
                    train_stats = {
                        "train/kd_loss": avg_kd_loss,
                        "train/kd_loss_std": torch.std(torch.tensor(batch_losses)).item() if len(batch_losses) > 1 else 0.0,
                        "train/update_counter": update_counter,
                        "train/updates_this_step": len(batch_losses),
                        "train/student_lr": avg_lr,
                        "train/gpu_memory_allocated": torch.cuda.memory_allocated(cfg.device) / 1e9,  # GB
                        "train/gpu_memory_reserved": torch.cuda.memory_reserved(cfg.device) / 1e9,   # GB
                    }
                    # 添加维度级损失到统计中
                    train_stats.update(avg_kd_losses)
                    info.update(train_stats)
                    
                    # 清理batch statistics
                    # del batch_losses, batch_lr_values
                
            else:
                # Buffer warmup phase
                info.update({
                    "train/status": "buffer_warmup",
                    "train/warmup_progress": len(replay_buffer) / warmup_steps
                })
            
            # Episode statistics
            episode_stats.add(data)
            if len(episode_stats) >= transformed_env.num_envs:
                stats = {
                    "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                    for k, v in episode_stats.pop().items(True, True)
                }
                info.update(stats)

            # Evaluation using student policy
            if i % cfg.eval_interval == 0 and i > 0:
                torch.cuda.empty_cache()
                print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating student policy at step: {i}")
                # env.enable_render(True)
                env.eval()
                
                # Evaluate student policy
                eval_info_student = evaluate(
                    env=transformed_env, 
                    policy=policy_s,
                    seed=cfg.seed, 
                    cfg=cfg,
                    exploration_type=ExplorationType.MEAN
                )
                
                # # Optionally compare with teacher performance
                # eval_info_teacher = evaluate(
                #     env=transformed_env, 
                #     policy=policy_t,
                #     seed=cfg.seed, 
                #     cfg=cfg,
                #     exploration_type=ExplorationType.MEAN
                # )
                
                # env.enable_render(not cfg.headless)
                env.train()
                env.reset()
                
                # Add prefixes to distinguish student vs teacher metrics
                eval_info = {}
                for k, v in eval_info_student.items():
                    eval_info[f"student_{k}"] = v
                # for k, v in eval_info_teacher.items():
                #     eval_info[f"teacher_{k}"] = v
                    
                info.update(eval_info)
                print(f"Student performance vs Teacher comparison logged")

            # Log to wandb
            run.log(info)
            
            # 定期显存清理
            if i % 10 == 0:
                torch.cuda.empty_cache()
            
            if i % 50 == 0:
                print(f"\n[Step {i}] RL-based Knowledge Distillation Training Summary:")
                print(f"  {'='*58}")
                for k, v in info.items():
                    if isinstance(v, (float, int)) and 'train/' in k:
                        if 'gpu_memory' in k:
                            print(f"  {k:<35}: {v:.2f} GB")
                        elif 'q_student' in k:
                            print(f"  {k:<35}: {v:.4f}  [Q-value from Teacher Critic]")
                        elif 'kd_rl_loss' in k:
                            print(f"  {k:<35}: {v:.4f}  [RL loss: -Q]")
                        elif 'kd_bc_loss' in k:
                            print(f"  {k:<35}: {v:.4f}  [BC loss: MSE]")
                        elif 'action_diff' in k:
                            print(f"  {k:<35}: {v:.4f}  [Action difference]")
                        else:
                            print(f"  {k:<35}: {v:.4f}")
                print(f"  {'='*58}")
                print(f"  Buffer Size: {len(replay_buffer):,}")
                print(f"  Total Updates: {update_counter:,}")
                if 'train/updates_this_step' in info:
                    print(f"  Updates/Step: {info['train/updates_this_step']}")
                print(f"  kd_alpha (RL weight): {kd_alpha}, bc_weight: {bc_weight}")
                print("-" * 60)

            # Save student model
            if i % cfg.save_interval == 0:
                ckpt_path = os.path.join(run.dir, f"student_checkpoint_{i}.pt")
                torch.save(policy_s.state_dict(), ckpt_path)
                print(f"[NavRL]: Student model saved at step: {i}")

        # Final save
        ckpt_path = os.path.join(run.dir, "student_checkpoint_final.pt")
        torch.save(policy_s.state_dict(), ckpt_path)
        wandb.finish()
        sim_app.close()
        
    except KeyboardInterrupt:
        print("\n[NavRL]: KeyboardInterrupt received. Saving student model...")
        ckpt_path = os.path.join(run.dir, f"student_checkpoint_interrupt.pt")
        torch.save(policy_s.state_dict(), ckpt_path)
        run.log({"interrupted_at_step": i})
        run.finish()
        sim_app.close()
        exit(0)

if __name__ == "__main__":
    main()