<!--
 * @Author: zdytim zdytim@foxmail.com
 * @Date: 2026-01-06 11:23:07
 * @LastEditors: zdytim zdytim@foxmail.com
 * @LastEditTime: 2026-01-06 11:23:08
 * @FilePath: /NavRL/isaac-training/training/scripts/models/SAC_REFACTORING_SUMMARY.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
"""
SAC 模型抽象化总结
==================

本文档说明了将 SAC 模型从训练脚本中抽象出来的工作。

作者: NavRL Team
日期: 2026年1月6日
"""

## 1. 改动概述

将 SAC (Soft Actor-Critic) 的模型定义从训练脚本中分离出来，放到独立的模型管理模块中，
遵循与 PPO-ViT 模型相同的设计模式。

## 2. 文件结构

### 新增文件

```
isaac-training/training/scripts/models/
├── sac_model.py                    # 🆕 SAC 模型管理模块
├── example_sac_usage.py            # 🆕 使用示例脚本
└── README.md                       # 📝 更新文档
```

### 修改文件

```
isaac-training/training/scripts/
└── SAC_v1.py                       # 📝 添加 SAC_V2 包装类
```

## 3. 主要组件

### 3.1 SACFeatureExtractor

共享特征提取器，包含：
- Lidar CNN 特征提取（Conv2D + ELU + LayerNorm）
- 动态障碍物编码器（MLP）
- 特征融合（拼接 + LayerNorm）
- TensorDict 接口支持

**输入：**
- `lidar`: [Batch, 1, 60, 60]
- `dynamic_obstacle`: [Batch, 1, 10, 5]
- `state`: [Batch, 8]

**输出：**
- `feature`: [Batch, 200] (128 + 64 + 8)

### 3.2 ActorNetwork

Actor 网络，输出 TanhNormal 分布参数：
- 特征提取器 → 200维特征
- MLP (200 → 256 → 256) + LayerNorm
- GaussianActor → (loc, scale)
- TanhNormal 分布包装

**输出：**
- `action_normalized`: 归一化动作 [-1, 1]
- `loc`: 分布均值
- `scale`: 分布标准差

### 3.3 CriticNetwork

Critic 网络（Q函数）：
- 特征提取器 → 200维特征
- 拼接动作 → 200 + action_dim
- MLP (200+act_dim → 256 → 256 → 1)
- 输出 Q 值

**输入：**
- `state`: 观测字典
- `action`: [Batch, action_dim]

**输出：**
- `q_value`: [Batch, 1]

### 3.4 SACModel

完整的 SAC 模型，包含：
- Actor 网络
- 双 Critic 网络（Q1, Q2）
- Target Critic 网络（Q1_target, Q2_target）
- Temperature 参数（log_alpha, alpha）
- 优化器（actor_optim, critic1_optim, critic2_optim, alpha_optim）
- 训练逻辑（train_step）
- 软更新逻辑（_soft_update）

**核心方法：**
- `get_action(state, deterministic)`: 推理接口
- `__call__(td)`: 环境交互接口
- `train_step(replay_buffer, batch_size, tau)`: 训练步骤
- `actions_to_world(actions, tensordict)`: 坐标转换

### 3.5 SACModelManager

模型管理器，提供统一接口：
- 模型创建和初始化
- 检查点保存 (`save_checkpoint`)
- 检查点加载 (`load_checkpoint`)
- wandb 集成（自动上传）
- 训练模式切换 (`set_training_mode`)
- 模型信息查询 (`get_model_info`)

## 4. 使用方式

### 4.1 方式一：直接使用 SACModelManager（推荐）

```python
from models.sac_model import SACModelManager

# 创建模型管理器
manager = SACModelManager(
    cfg=cfg,
    observation_spec=env.observation_spec,
    action_spec=env.action_spec,
    device=device
)

# 推理
actions = manager.get_action(observations, deterministic=True)

# 训练
loss_info = manager.train_step(replay_buffer, batch_size=256, tau=0.005)

# 保存/加载
manager.save_checkpoint("checkpoint.pt", step=10000)
manager.load_checkpoint("checkpoint.pt")
```

### 4.2 方式二：使用 SAC_V2 包装类（向后兼容）

```python
from SAC_v1 import SAC_V2

# 创建 SAC agent（接口与原始 SAC 相同）
agent = SAC_V2(cfg, obs_spec, act_spec, device)

# 使用方式与原始 SAC 完全相同
agent.get_action(state)
agent.train(replay_buffer, batch_size)
agent.save_checkpoint(path, step)
```

### 4.3 方式三：便捷函数

```python
from models.sac_model import create_sac_model

# 一行创建
manager = create_sac_model(cfg, obs_spec, act_spec, device)
```

## 5. 设计优势

### 5.1 关注点分离
- **模型定义** → `models/sac_model.py`
- **训练逻辑** → `train_sac.py` (训练脚本)
- **配置管理** → `cfg/*.yaml`

### 5.2 代码复用
- 特征提取器在 Actor 和 Critic 间共享架构
- 统一的检查点管理接口
- wandb 集成开箱即用

### 5.3 易于扩展
- 新增模型只需继承 `TensorDictModuleBase`
- 实现 `get_action`, `train_step`, `__call__` 方法
- 管理器自动处理检查点和配置

### 5.4 向后兼容
- 保留原始 SAC 类（标记为废弃）
- 提供 SAC_V2 包装类
- 接口完全一致，无需修改现有代码

## 6. 检查点格式

保存的检查点包含：
```python
{
    'step': int,                              # 训练步数
    'model_state_dict': OrderedDict,          # 模型参数
    'actor_optim_state_dict': dict,           # Actor 优化器
    'critic1_optim_state_dict': dict,         # Critic1 优化器
    'critic2_optim_state_dict': dict,         # Critic2 优化器
    'alpha_optim_state_dict': dict,           # Temperature 优化器
    'log_alpha': Tensor,                      # log(alpha) 参数
    'cfg': dict,                              # 配置信息
    **extra_info                              # 额外信息（如 replay_buffer）
}
```

## 7. wandb 集成

如果 wandb 可用且已初始化，`save_checkpoint` 会自动：
1. 保存检查点到本地
2. 创建 wandb Artifact（类型：model）
3. 上传到 wandb 云端
4. 关联到当前 run

禁用 wandb：
```python
import wandb
wandb.init(mode="disabled")
```

## 8. 迁移指南

### 步骤 1: 更新导入

**旧代码：**
```python
from SAC_v1 import SAC
```

**新代码：**
```python
from SAC_v1 import SAC_V2  # 或
from models.sac_model import SACModelManager
```

### 步骤 2: 创建模型

**旧代码：**
```python
sac_agent = SAC(cfg, obs_spec, act_spec, device)
```

**新代码：**
```python
# 方式1（推荐）
manager = SACModelManager(cfg, obs_spec, act_spec, device)

# 方式2（兼容）
sac_agent = SAC_V2(cfg, obs_spec, act_spec, device)
```

### 步骤 3: 其他代码无需修改

所有方法接口保持一致：
- `get_action(state, deterministic)`
- `train(replay_buffer, batch_size, tau)`
- `save_checkpoint(path, step, **extra)`
- `load_checkpoint(path, load_optimizers)`

## 9. 测试和验证

### 运行示例脚本

```bash
cd isaac-training/training/scripts/models
python example_sac_usage.py
```

### 验证项目
- ✅ 模型创建和初始化
- ✅ 推理（确定性和随机）
- ✅ 检查点保存和加载
- ✅ 向后兼容性
- ⚠️  训练步骤（需要真实 ReplayBuffer）

## 10. 未来改进

- [ ] 添加混合精度训练支持（AMP）
- [ ] 支持分布式训练
- [ ] 添加模型剪枝和量化接口
- [ ] 实现自动超参数调优
- [ ] 支持更多 SAC 变体（如 SAC-Discrete）

## 11. 参考资料

- TorchRL 文档: https://pytorch.org/rl/
- TensorDict 文档: https://github.com/pytorch/tensordict
- SAC 论文: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"
- NavRL PPO-ViT 模型: `models/navrl_model.py`

---

**总结：** 本次重构将 SAC 模型完全抽象化，提供了统一、可扩展、易维护的模型管理接口，
同时保持向后兼容性。训练脚本现在只需关注采样、训练循环和日志记录等高层逻辑。
