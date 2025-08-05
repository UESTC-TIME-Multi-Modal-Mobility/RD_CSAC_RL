'''
Author: zdytim zdytim@foxmail.com
Date: 2025-07-28 23:29:23
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2025-08-03 18:47:26
FilePath: /u20/NavRL/isaac-training/training/scripts/SAC_train.py
Description: SAC_算法训练
'''
import argparse
import warnings
import os
import hydra
import datetime
import wandb
import torch
import torch.cuda
import tqdm
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
# from torch.cuda.amp import autocast, GradScaler
from utils import evaluate
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl._utils import compile_with_warmup,timeit
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from tensordict import TensorDict
# from tensordict.nn import CudaGraphModule
import datetime
from SAC_utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
)

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    #仿真加载
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    #日志加载
    # Use Wandb to monitor training
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            # entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            # entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )
    device = torch.device(cfg.device)
    #环境加载
    # Navigation Training Environment
    from env import NavigationEnv
    env = NavigationEnv(cfg)
    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)
    eval_env = transformed_env
    #决策模型加载
    model, exploration_policy = make_sac_agent(cfg,transformed_env, eval_env, device)
    # SAC Policy
    # Create SAC loss
    loss_module, target_net_updater = make_loss_module(cfg, model)
    # 动态设置编译模式：根据配置文件和其他参数，选择合适的编译模式（如 "default" 或 "reduce-overhead"）。
    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"
    # policy = SACLoss(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    policy = exploration_policy
    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    # checkpoint = "/home/u20/NavRL/isaac-training/training/scripts/wandb/offline-run-20250716_112035-mnnbcqxu/files/checkpoint_46000.pt"
    # policy.load_state_dict(torch.load(checkpoint))

    #数据收集
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)
    # RL Data Collector
    collector = SyncDataCollector(
        transformed_env,
        policy=exploration_policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )
    #建立经验缓存池
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=cfg.device,
    )
    
    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_sac_optimizer(cfg, loss_module)
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
    del optimizer_actor, optimizer_critic, optimizer_alpha

    def update(sampled_tensordict):
        # Compute loss
        loss_td = loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        (actor_loss + q_loss + alpha_loss).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Update qnet_target params
        target_net_updater.step()
        return loss_td.detach()

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)

    # if cfg.compile.cudagraphs:
    #     warnings.warn(
    #         "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
    #         category=UserWarning,
    #     )
    #     update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    # Main loop
    collected_frames = 0 #已经收集的总帧数
    #表示训练过程中需要收集的总帧数。
    pbar = tqdm.tqdm(total=cfg.max_frame_num) #一个进度条，用于显示训练过程中收集的帧数进度
    #在策略网络尚未训练时，使用随机动作填充经验回放缓冲区
    init_random_frames = cfg.collector.init_random_frames # 在训练开始时，使用随机动作收集的初始帧数
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)#每次从 Replay Buffer 中采样并更新模型的次数。
    prb = cfg.replay_buffer.prb #是否启用优先经验回放（Prioritized Replay Buffer）
    eval_iter = cfg.eval_interval #评估间隔，表示每隔多少帧进行一次评估
    frames_per_batch = cfg.collector.frames_per_batch #每次从环境中收集的帧数
    eval_rollout_steps = cfg.env.max_episode_steps #评估时每次 rollout 的最大步数。
    collector_iter = iter(collector) #数据收集器的迭代器
    total_iter = int(cfg.collector.total_frames // cfg.collector.frames_per_batch)# 数据收集器的总迭代次数
    #
    for i in range(total_iter):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)
        # info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        with timeit("collect"):
            tensordict = next(collector_iter)
        # Update weights of the inference policy
        collector.update_policy_weights_()
        # Update episode stats
        current_frames = tensordict.numel()
        pbar.update(current_frames)
        
        with timeit("rb - extend"):
            # Add to replay buffer
            tensordict = tensordict.reshape(-1)
            replay_buffer.extend(tensordict)
            
        collected_frames += current_frames
        
        # Optimization steps
        with timeit("train"):
            if collected_frames >= init_random_frames:
                losses = TensorDict(batch_size=[num_updates])
                for i in range(num_updates):
                    with timeit("rb - sample"):
                        # Sample from replay buffer
                        sampled_tensordict = replay_buffer.sample()

                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss_td = update(sampled_tensordict).clone() #train_loss_stats = policy.train(data)
                    losses[i] = loss_td.select(
                        "loss_actor", "loss_qvalue", "loss_alpha"
                    )
                    # Update priority
                    if prb:
                        replay_buffer.update_priority(sampled_tensordict)
        # 统计 Episode 奖励
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]
        
        # Logging
        # metrics_to_log = {}
        # if len(episode_rewards) > 0:
        #     episode_length = tensordict["next", "step_count"][episode_end]
        #     metrics_to_log["train/reward"] = episode_rewards
        #     metrics_to_log["train/episode_length"] = episode_length.sum() / len(
        #         episode_length
        #     )
        # if collected_frames >= init_random_frames:
        #     losses = losses.mean()
        #     metrics_to_log["train/q_loss"] = losses.get("loss_qvalue")
        #     metrics_to_log["train/actor_loss"] = losses.get("loss_actor")
        #     metrics_to_log["train/alpha_loss"] = losses.get("loss_alpha")
        #     metrics_to_log["train/alpha"] = loss_td["alpha"]
        #     metrics_to_log["train/entropy"] = loss_td["entropy"]
        
        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(
                ExplorationType.DETERMINISTIC
            ), torch.no_grad(), timeit("eval"):
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
        # Update wand info
        run.log(info)
        print(f"\n[Step {i}] Training Summary:")
        for k, v in info.items():
            if isinstance(v, (float, int)):
                print(f"  {k:<20}: {v:.4f}")
        print("-" * 40)
        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(model[0], ckpt_path)
            print("[NavRL]: model saved at training step: ", i)
    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(model[0], ckpt_path)
    wandb.finish()
    sim_app.close()
                # eval_rollout = eval_env.rollout(
                #     eval_rollout_steps,
                #     model[0],
                #     auto_cast_to_device=True,
                #     break_when_any_done=True,
                # )
                # eval_env.apply(dump_video)
                # eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                # metrics_to_log["eval/reward"] = eval_reward
        # if logger is not None:
        #     metrics_to_log.update(timeit.todict(prefix="time"))
        #     metrics_to_log["time/speed"] = pbar.format_dict["rate"]
        #     log_metrics(logger, metrics_to_log, collected_frames)
        
if __name__ == "__main__":
    main()
