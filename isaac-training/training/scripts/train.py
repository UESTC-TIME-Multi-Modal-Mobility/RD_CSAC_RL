import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from SAC import SAC
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
# from torch.cuda.amp import autocast, GradScaler
from utils import evaluate
from torchrl.envs.utils import ExplorationType
import datetime

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Use Wandb to monitor training
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            # entity=cfg.wandb.entity,
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            # entity=cfg.wandb.entity,
            config=wandb_config,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # Navigation Training Environment
    from env import NavigationEnv
    env = NavigationEnv(cfg)
    # scaler = GradScaler()
    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)
    
    # SAC Policy
    policy = SAC(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    
    # Replay Buffer for SAC
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.algo.buffer_size),
        batch_size=cfg.algo.batch_size if hasattr(cfg.algo, 'batch_size') else 128,
    )

    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    # checkpoint = "/home/u20/NavRL/isaac-training/training/scripts/wandb/offline-run-20250716_112035-mnnbcqxu/files/checkpoint_46000.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector for SAC
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True,
        exploration_type=ExplorationType.RANDOM,
    )
    
    # SAC specific variables
    update_counter = 0
    warmup_steps = cfg.algo.warmup_steps if hasattr(cfg.algo, 'warmup_steps') else 1000
    try:
    # Training Loop for SAC
        for i, data in enumerate(collector):
            # Add data to replay buffer
            #TODO：这个地方有问题，没有添加动作与下一刻的状态，应该统一
            replay_buffer.extend(data)
            
            # Log Info
            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            
            # Start training only after warmup
            if replay_buffer.__len__() >= warmup_steps:
                # Train Policy with SAC
                train_loss_stats = policy.train(replay_buffer)
                info.update(train_loss_stats)
                update_counter += 1
            else:
                info.update({"status": "warming_up", "buffer_size": replay_buffer.__len__()})
            
            # Calculate and log training episode stats
            episode_stats.add(data)
            if len(episode_stats) >= transformed_env.num_envs:
                stats = {
                    "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                    for k, v in episode_stats.pop().items(True, True)
                }
                info.update(stats)

            # Evaluate policy and log info
            if i % cfg.eval_interval == 0:
                print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] start evaluating policy at training step:{i}")
                # print("[NavRL]: start evaluating policy at training step: ", i)
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
                # print("\n[NavRL]: evaluation done.")
                # torch.cuda.synchronize()
                # torch.cuda.empty_cache()
                print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [NavRL]: evaluation done.")
            
            # Update wandb info
            run.log(info)
            print(f"\n[Step {i}] SAC Training Summary:")
            for k, v in info.items():
                if isinstance(v, (float, int)):
                    print(f"  {k:<20}: {v:.4f}")
            print(f"  Buffer Size: {replay_buffer.__len__()}")
            print(f"  Updates: {update_counter}")
            print("-" * 40)
            

            # Save Model
            if i % cfg.save_interval == 0:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                print("[NavRL]: model saved at training step: ", i)

        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)
        wandb.finish()
        sim_app.close()
    except KeyboardInterrupt:
        print("\n[NavRL]: KeyboardInterrupt received. Saving model and finishing wandb run.")
        ckpt_path = os.path.join(run.dir, f"checkpoint_interrupt.pt")
        torch.save(policy.state_dict(), ckpt_path)
        run.log({"interrupted_at_step": i})
        run.finish()
        sim_app.close()
        exit(0)

if __name__ == "__main__":
    main()