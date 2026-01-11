import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import PPO
# from SAC_v1 import SAC
# from ppo_vit_v3 import PPOVIT
from models.navrl_model import NavRLModel
# from SAC_lag import SAC
from models.sac_model import SACModelManager, SACModel
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../cfg")
@hydra.main(config_path=FILE_PATH, config_name="eval", version_base=None)
def main(cfg):
    # Simulation App
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Use Wandb to monitor training
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

    # Transformed Environment
    transforms = []
    # transforms.append(ravel_composite(env.observation_spec, ("agents", "intrinsics"), start_dim=-1))
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)
    # PPO Policy
    # policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    # policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    # policy = SAC(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    print(f"[NavRL]: action_spec: {transformed_env.action_spec}")
    policy = NavRLModel(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
    checkpoint_paths = [
        os.path.join(checkpoint_dir, fname)
        for fname in os.listdir(checkpoint_dir)
        if fname.endswith(".pt")
    ]
    checkpoint_paths.sort()
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoint files (*.pt) found in {checkpoint_dir}."
        )
    
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # RL Data Collector

    eval_seed = cfg.seed

    # Evaluation Loop
    for ckpt_path in checkpoint_paths:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"[NavRL]: start evaluating policy from {ckpt_name} with seed {eval_seed}")

        state_dict = torch.load(ckpt_path, map_location=cfg.device)
        policy.load_state_dict(state_dict)

        info = {
            "env_seed": eval_seed,
            "checkpoint": ckpt_name,
        }
        env.enable_render(True)
        env.eval()
        transformed_env.set_seed(eval_seed)
        eval_info = evaluate(
            env=transformed_env,
            policy=policy,
            seed=eval_seed,
            cfg=cfg,
            exploration_type=ExplorationType.MEAN,
        )
        env.enable_render(not cfg.headless)
        env.train()
        env.reset()
        info.update(eval_info)

        print("\n[NavRL]: evaluation done.")

        # Update wandb info
        run.log(info)
        print(f"[NavRL]: eval info: {info}")


        # # Save Model
        # if i % cfg.save_interval == 0:
        #     ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
        #     torch.save(policy.state_dict(), ckpt_path)
        #     print("[NavRL]: model saved at training step: ", i)

    # ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    # torch.save(policy.state_dict(), ckpt_path)
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    