import argparse
import os
import hydra
import datetime
import wandb
import torch
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
from utils import evaluate
from torchrl.envs.utils import ExplorationType
import datetime

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../cfg")

# def knowledge_distillation_loss(student_output, teacher_output, temperature=3.0, alpha=0.7):
#     """
#     NavRL知识蒸馏损失计算，适配连续动作空间
#     Args:
#         student_output: tuple (actions, mu, log_std) from student actor
#         teacher_output: tuple (actions, mu, log_std) from teacher actor  
#         temperature: 蒸馏温度，控制分布软化程度
#         alpha: 蒸馏损失权重
#     """
#     if isinstance(student_output, tuple) and isinstance(teacher_output, tuple):
#         # SAC actor返回tuple: (action, mu, log_std)
#         student_action, student_mu, student_log_std = student_output
#         teacher_action, teacher_mu, teacher_log_std = teacher_output
        
#         # 对连续动作使用MSE损失
#         action_loss = F.mse_loss(student_action, teacher_action.detach())
#         mu_loss = F.mse_loss(student_mu, teacher_mu.detach()) 
#         std_loss = F.mse_loss(student_log_std, teacher_log_std.detach())
        
#         distillation_loss = alpha * (action_loss + 0.5 * (mu_loss + std_loss))
#     else:
#         # fallback for tensor inputs
#         distillation_loss = alpha * F.mse_loss(student_output, teacher_output.detach())
    
#     return distillation_loss
def knowledge_distillation_loss(student_actions, teacher_actions, temperature=3.0, alpha=0.7):
    """
    NavRL知识蒸馏损失计算，适配连续动作空间和TensorDict结构
    对三个维度(vx, vy, vz)分别计算损失以便于监控
    Args:
        student_actions: Tensor [batch_size, 3] from student policy normalized actions
        teacher_actions: Tensor [batch_size, 3] from teacher policy normalized actions
        temperature: 蒸馏温度，控制分布软化程度
        alpha: 蒸馏损失权重
    Returns:
        total_loss: 总的蒸馏损失
        loss_dict: 包含各维度损失的字典，用于wandb记录
    """
    # NavRL uses normalized actions in [-1, 1] range before world coordinate transform
    # Actions represent [vx, vy, vz] in goal-aligned frame via utils.vec_to_world
    
    # 分别计算三个维度的MSE损失
    loss_vx = F.mse_loss(student_actions[:, 0], teacher_actions[:, 0].detach())
    loss_vy = F.mse_loss(student_actions[:, 1], teacher_actions[:, 1].detach())
    loss_vz = F.mse_loss(student_actions[:, 2], teacher_actions[:, 2].detach())
    
    # 总损失（加权平均）
    total_loss = alpha * (loss_vx + loss_vy + loss_vz) / 3.0
    
    # 返回详细的损失字典供wandb记录
    loss_dict = {
        "kd_loss_vx": loss_vx.item(),
        "kd_loss_vy": loss_vy.item(),
        "kd_loss_vz": loss_vz.item(),
        "kd_loss_total": total_loss.item(),
        # 记录各维度的相对重要性
        "kd_loss_vx_ratio": (loss_vx / (loss_vx + loss_vy + loss_vz + 1e-8)).item(),
        "kd_loss_vy_ratio": (loss_vy / (loss_vx + loss_vy + loss_vz + 1e-8)).item(),
        "kd_loss_vz_ratio": (loss_vz / (loss_vx + loss_vy + loss_vz + 1e-8)).item(),
    }
    
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
    
    # Load pretrained models
    policy_t.load_state_dict(torch.load("/home/u20/NavRL/isaac-training/training/scripts/checkpoint/CSAC_0_2100.pt"))
    # Set training modes properly - avoid calling .eval() on SAC due to method name conflict
    # Teacher should not be updated, so set requires_grad=False for all parameters
    for param in policy_t.parameters():
        param.requires_grad = False
    policy_t.training = False  # Set training flag directly instead of calling .eval()

    # policy_s.load_state_dict(torch.load("/home/u20/NavRL/isaac-training/training/scripts/checkpoint/student_checkpoint_50000.pt"))
    # Student remains in training mode
    policy_s.training = True
    for param in policy_s.parameters():
        param.requires_grad = True

    # Replay buffer for storing teacher-generated data
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.algo.buffer_size,  # 减少buffer大小节省显存
            # device="cpu"  # 将buffer存储在CPU上
        ),
        batch_size=cfg.algo.batch_size,  # 减少batch size
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
        policy=policy_t,  # Use teacher policy for data collection
        frames_per_batch=cfg.env.num_envs * cfg.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True,
        exploration_type=ExplorationType.MEAN,  # Use teacher's mean action
    )

    # Knowledge distillation parameters
    kd_temperature = cfg.get('kd_temperature', 3.0)
    kd_alpha = cfg.get('kd_alpha', 0.7)
    update_counter = 0
    warmup_steps = cfg.algo.warmup_steps
    
    # 增加训练频次参数 - 提高样本利用效率
    training_epoch_num = cfg.get('training_epoch_num', 3)  # 减少到2次，避免显存溢出
    print(f"[NavRL KD]: training_epoch_num: {training_epoch_num}")
    print(f"[NavRL KD]: Reduced for memory efficiency with 224x224 inputs")

    # # 显存优化设置
    # torch.backends.cudnn.benchmark = False  # 减少显存使用
    # torch.cuda.empty_cache()  # 清理显存缓存



    try:
        for i, data in enumerate(collector):
            # Store teacher-generated data
            data = data.reshape(-1).cpu()
            replay_buffer.extend(data)
            
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            
            if len(replay_buffer) >= warmup_steps:
                # 对每批数据进行多次训练更新 - 显存优化版本
                batch_losses = []
                batch_lr_values = []
                
                for update_idx in range(training_epoch_num):
                    # 显存管理 - 每次更新前清理
                    # torch.cuda.empty_cache()
                    
                    # Sample batch from teacher experience buffer
                    batch = replay_buffer.sample().to(cfg.device)
                    
                    obs = batch[("agents", "observation")]
                    
                    # Teacher inference with frozen parameters (no gradients)
                    with torch.no_grad():
                        # 使用更简洁的teacher调用，避免额外TensorDict创建
                        teacher_output = policy_t(batch)
                        teacher_actions = teacher_output[("agents", "action_normalized")]
                        
                        # 立即释放teacher相关tensor
                        del teacher_output
                    
                    # Student forward pass - 内存优化
                    with torch.cuda.amp.autocast(enabled=True):  # 使用混合精度
                        # 直接使用batch而不是重新创建TensorDict
                        student_latent = policy_s.shared_features(
                            obs["camera"],
                            obs["dynamic_obstacle"], 
                            obs["state"]
                        )
                        
                        # 创建临时tensordict for actor
                        temp_td = batch.clone()
                        temp_td.set("_latent", student_latent)
                        
                        # Get student actions
                        policy_s.actor_head(temp_td)
                        student_actions = temp_td[("agents", "action_normalized")]
                        
                        # Knowledge distillation loss with dimension-wise tracking
                        kd_loss, kd_loss_dict = knowledge_distillation_loss(
                            student_actions, teacher_actions, 
                            temperature=kd_temperature, alpha=kd_alpha
                        )
                    
                    # 清理中间变量
                    # del student_latent, temp_td
                    
                    # Student parameter update - 混合精度优化
                    policy_s.optimizer.zero_grad()
                    
                    if hasattr(policy_s, 'scaler') and policy_s.scaler:
                        policy_s.scaler.scale(kd_loss).backward()
                        policy_s.scaler.unscale_(policy_s.optimizer)
                        torch.nn.utils.clip_grad_norm_(policy_s.parameters(), max_norm=1.0)
                        policy_s.scaler.step(policy_s.optimizer)
                        policy_s.scaler.update()
                    else:
                        kd_loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_s.parameters(), max_norm=1.0)
                        policy_s.optimizer.step()
                    
                    # 收集统计信息（包含维度级损失）
                    batch_losses.append(kd_loss.item())
                    batch_lr_values.append(policy_s.optimizer.param_groups[0]['lr'])
                    
                    # 累积维度级损失用于平均
                    if update_idx == 0:
                        batch_kd_losses = {k: [v] for k, v in kd_loss_dict.items()}
                    else:
                        for k, v in kd_loss_dict.items():
                            batch_kd_losses[k].append(v)
                    
                    update_counter += 1
                    
                    # # 立即清理本次更新的tensor
                    # del batch, obs, student_actions, teacher_actions, kd_loss
                    # torch.cuda.empty_cache()
                
                # 计算平均损失和训练统计
                if batch_losses:  # 确保有有效的损失记录
                    avg_kd_loss = sum(batch_losses) / len(batch_losses)
                    avg_lr = sum(batch_lr_values) / len(batch_lr_values)
                    
                    # 计算维度级损失的平均值
                    avg_kd_losses = {
                        f"train/{k}": sum(v) / len(v) 
                        for k, v in batch_kd_losses.items()
                    }
                    
                    # Log training statistics with memory usage and dimension-wise losses
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
                print(f"\n[Step {i}] Knowledge Distillation Training Summary:")
                for k, v in info.items():
                    if isinstance(v, (float, int)) and 'train/' in k:
                        if 'gpu_memory' in k:
                            print(f"  {k:<30}: {v:.2f} GB")
                        else:
                            print(f"  {k:<30}: {v:.4f}")
                print(f"  Buffer Size: {len(replay_buffer):,}")
                print(f"  Total Updates: {update_counter:,}")
                if 'train/updates_this_step' in info:
                    print(f"  Updates/Step: {info['train/updates_this_step']}")
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