import torch
import torch.nn as nn
import wandb
import numpy as np
import pandas as pd
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        MAX_BETA_PARAM = 20.0  # 或者10.0，根据实际情况调整
        alpha = torch.clamp(alpha, min=1.0001, max=MAX_BETA_PARAM)
        beta = torch.clamp(beta, min=1.0001, max=MAX_BETA_PARAM)
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta


class GaussianActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.mean_layer = nn.Linear(256, action_dim)
        # 使用形状为 (1, action_dim) 的参数，便于在 forward 时按 batch 扩展
        self.log_std_param = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, features: torch.Tensor):
        # features: [B, 256]（或兼容形状）
        loc = self.mean_layer(features)  # [B, action_dim]
        # clamp 防止 log_std 漂移到极端值
        log_std = self.log_std_param.clamp(-20.0, 2.0)
        # 按 batch 扩展到与 loc 匹配的 shape
        log_std = log_std.expand_as(loc)  # [B, action_dim]
        return loc, log_std

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    total_samples = tensordict.shape[0]
    
    # 确保至少有足够的样本来创建minibatches
    if total_samples < num_minibatches:
        print(f"Warning: total_samples ({total_samples}) < num_minibatches ({num_minibatches}), adjusting to {total_samples}")
        num_minibatches = max(1, total_samples)
    
    # 调整样本数量以确保可以整除
    samples_per_batch = total_samples // num_minibatches
    if samples_per_batch == 0:
        # 如果每个批次都没有样本，返回整个数据集作为单个批次
        yield tensordict
        return
        
    usable_samples = samples_per_batch * num_minibatches
    
    perm = torch.randperm(usable_samples, device=tensordict.device).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MEAN
):

    # 禁用渲染以节省显存，只收集数据
    env.enable_render(True)
    env.eval()
    env.set_seed(seed)

    render_callback = RenderCallback(interval=2)
    
    with set_exploration_type(exploration_type):
        trajs = env.rollout(
            max_steps=env.max_episode_length,
            policy=policy,
            callback=render_callback, 
            # callback=None,# 禁用视频录制
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        )
    # base_env.enable_render(not cfg.headless)
    env.enable_render(not cfg.headless)  # 保持渲染关闭状态
    env.reset()
    
    done = trajs.get(("next", "done")) 
    first_done = torch.argmax(done.long(), dim=1).cpu() # idx of first done will be return for each trajs

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    info = {
        "eval/stats." + k: torch.mean(v.float()).item() 
        for k, v in traj_stats.items()
    }

    # 禁用视频录制以节省显存
    info["recording"] = wandb.Video(
        render_callback.get_video_array(axes="t c h w"), 
        fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), 
        format="mp4"
    )
    
    env.train()
    # env.reset()

    return info
# @torch.no_grad()
# def evaluate(
#     env,
#     policy,
#     cfg,
#     seed: int=0, 
#     exploration_type: ExplorationType=ExplorationType.MEAN
# ):
#     print(f"\n[NavRL Eval]: 🟢 Starting Memory-Efficient Evaluation (Seed {seed})...")
    
#     # 1. 强制 Train 模式 (开启并行物理)
#     env.enable_render(False) # 彻底关闭渲染接口
#     env.train()  
    
#     # 2. 策略设为 Eval (确定性)
#     if hasattr(policy, "eval"):
#         policy.eval()

#     env.set_seed(seed)
    
#     # 3. 重置环境，获取初始观测
#     print("[NavRL Eval]: Resetting environment...")
#     tensordict = env.reset()
    
#     # 4. 初始化统计容器
#     # 我们只记录每个环境"第一次"完成任务时的数据，避免重复统计
#     finished_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
#     final_stats = {} 
    
#     # 5. 手动循环 (替代 env.rollout)
#     # 这样我们可以控制每一步都不保存历史图像，只保存统计数据
#     max_steps = 12000 # 只要时间够长，就能飞完
#     print(f"[NavRL Eval]: Running loop for {max_steps} steps (Discarding history)...")
    
#     import time
#     start_time = time.time()
    
#     for step in range(max_steps):
#         # A. 策略推理 (不保存梯度)
#         with set_exploration_type(exploration_type):
#             tensordict = policy(tensordict)
        
#         # B. 环境步进
#         tensordict = env.step(tensordict)
        
#         # C. 提取 Next State
#         tensordict = tensordict["next"]
        
#         # D. 实时统计 (关键步骤)
#         # 获取 done 信号 (terminated 或 truncated)
#         done = tensordict["done"].squeeze(-1) # [Num_Envs]
        
#         # 如果有环境刚刚完成 (done=True) 且之前没完成过
#         newly_finished = done & (~finished_mask)
        
#         if newly_finished.any():
#             # 提取这些环境的统计数据 (stats 存在于 tensordict 中)
#             # 注意：env.py 在 reset 时会清空 stats，所以要在 done 的这一帧抓取
#             current_stats = tensordict["stats"] # [Num_Envs, Stats_Dim]或其他结构
            
#             # 初始化 final_stats (如果是第一次)
#             if not final_stats:
#                 for k in current_stats.keys():
#                     # 预分配空间，避免碎片
#                     final_stats[k] = torch.zeros(env.num_envs, device=env.device)
            
#             # 记录数据
#             indices = newly_finished.nonzero().squeeze(-1)
#             for k, v in current_stats.items():
#                 # v 可能是 [Num_Envs, 1] 或 [Num_Envs]
#                 val = v[indices]
#                 if val.dim() > 1: val = val.squeeze(-1)
#                 final_stats[k][indices] = val
            
#             # 更新掩码
#             finished_mask = finished_mask | newly_finished
            
#             # 打印进度 (每完成 10% 打印一次)
#             completed_count = finished_mask.sum().item()
#             if step % 100 == 0:
#                  print(f"\r[Eval Progress]: Step {step}/{max_steps} | Completed: {completed_count}/{env.num_envs}", end="")

#         # E. 极其重要：处理 Auto-Reset
#         # IsaacEnv 通常会自动 reset，但我们需要确保 tensordict 里的 observation 是最新的
#         # 如果 env.step 内部处理了 reset，tensordict["next"] 已经是 reset 后的状态了
#         # 我们不需要手动 reset，只需要把 done 的环境标记一下即可
        
#         # F. 提前退出机制
#         if finished_mask.all():
#             print(f"\n[NavRL Eval]: All {env.num_envs} environments finished at step {step}!")
#             break
            
#     print(f"\n[NavRL Eval]: Loop finished. Duration: {time.time() - start_time:.2f}s")
    
#     # 6. 计算最终平均值
#     # 注意：只统计那些实际完成了的环境 (finished_mask)
#     # 如果没跑完 (例如 crash 了或者时间不够)，就只算跑完的
#     num_finished = finished_mask.sum().item()
#     if num_finished == 0:
#         print("[NavRL Eval]: ⚠️ WARNING: No environments finished! Check max_steps or difficulty.")
#         return {}

#     info = {}
#     for k, v in final_stats.items():
#         # 只取 finished 的部分求平均
#         valid_values = v[finished_mask]
#         info["eval/stats." + k] = torch.mean(valid_values.float()).item()

#     # 恢复 Policy 状态
#     if hasattr(policy, "train"):
#         policy.train()

#     print(f"[NavRL Eval]: Stats collected: {info}")
#     return info

def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


def construct_input(start, end):
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"

