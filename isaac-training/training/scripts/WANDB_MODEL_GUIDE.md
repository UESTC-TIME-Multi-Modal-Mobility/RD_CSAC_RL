# NavRL + wandb 模型管理完整指南

## 📊 wandb 模型管理功能概述

wandb 支持强大的模型版本管理和跟踪功能，NavRL 已完全集成以下功能：

### 🚀 核心功能

1. **模型 Artifacts**: 自动版本化模型文件
2. **模型注册表**: 生产级模型生命周期管理
3. **性能跟踪**: 训练指标与模型版本关联
4. **智能上传**: 基于性能阈值的智能模型保存
5. **便捷恢复**: 从 wandb 直接加载模型进行训练或推理

## 🔧 快速开始

### 1. 基础使用 - 集成到现有训练脚本

```python
# 导入 wandb 模型工具
from wandb_model_utils import upload_model_to_wandb, save_and_upload_best_model

# 在原有的模型保存代码后添加
torch.save(policy.state_dict(), ckpt_path)

# 上传到 wandb
upload_model_to_wandb(
    model_state_dict=policy.state_dict(),
    step=training_step,
    model_alias="latest"
)
```

### 2. 智能最佳模型保存

```python
# 评估后自动保存最佳模型
best_tracker = {'best_value': float('-inf'), 'best_step': 0}

eval_metrics = {"mean_reward": 85.2, "success_rate": 0.95}
is_new_best = save_and_upload_best_model(
    model_state_dict=policy.state_dict(),
    step=training_step,
    eval_metrics=eval_metrics,
    threshold_metric="mean_reward",
    best_value_tracker=best_tracker
)
```

### 3. 从 wandb 恢复训练

```python
# 从 wandb artifact 加载模型
from wandb_model_utils import download_model_from_wandb

model_path = download_model_from_wandb("username/project/model-name:best")
if model_path:
    policy.load_state_dict(torch.load(model_path))
    print("✅ Model restored from wandb")
```

## 📁 文件结构

```
isaac-training/training/scripts/
├── models/                          # 新的模型管理器
│   ├── __init__.py
│   ├── navrl_model.py              # 主要模型管理器 (包含 wandb 集成)
│   └── README.md
├── wandb_model_utils.py            # wandb 工具函数 (轻量级集成)
├── train_ppo_with_wandb.py         # 集成版训练脚本
├── train_with_wandb.py             # 完整的 wandb 训练器
└── cfg/
    └── wandb_model_config.yaml     # wandb 配置示例
```

## 🎯 使用场景

### 场景 1: 轻量级集成 (推荐用于现有项目)

**适用于**: 已有训练脚本，希望最小改动集成模型管理

```bash
# 1. 导入工具函数
from wandb_model_utils import upload_model_to_wandb

# 2. 在现有保存代码后添加上传
torch.save(model.state_dict(), path)
upload_model_to_wandb(model.state_dict(), step=step)

# 3. 运行训练
python train_ppo.py  # 你的现有训练脚本
```

### 场景 2: 完整集成 (推荐用于新项目)

**适用于**: 新项目或愿意重构训练代码

```bash
# 1. 使用新的训练脚本
python train_ppo_with_wandb.py

# 2. 或使用完整的训练器
python train_with_wandb.py

# 3. 从 wandb 恢复训练
python train_ppo_with_wandb.py resume_checkpoint="user/project/model:best"
```

### 场景 3: 使用新模型管理器

**适用于**: 希望更好的代码结构和功能

```python
from models import create_navrl_model

# 创建模型管理器
model_manager = create_navrl_model(cfg, obs_spec, act_spec, device)

# 保存并上传模型
model_manager.save_checkpoint(
    filepath="checkpoint.pt",
    step=1000,
    upload_to_wandb=True,
    wandb_alias="best"
)

# 从 wandb 加载
model_manager.load_from_wandb("user/project/model:latest")
```

## ⚙️ 配置选项

### 基础 wandb 配置

```yaml
wandb:
  project: "navrl-models"
  name: "ppo-vit-experiment"
  mode: "online"  # "offline", "disabled"

model_management:
  upload_frequency: 5          # 每5次保存上传一次
  save_best_models: true       # 自动保存最佳模型
  best_model_metric: "mean_reward"
```

### 高级配置

参考 `cfg/wandb_model_config.yaml` 了解完整配置选项。

## 📊 wandb 界面功能

### 1. Artifacts 面板
- 📁 **模型版本**: 所有训练过程中保存的模型版本
- 🏷️ **别名管理**: latest, best, stable 等版本标签
- 📈 **元数据**: 参数数量、训练步数、性能指标
- 📋 **模型卡片**: 自动生成的模型文档

### 2. 模型注册表
- 🎯 **生产模型**: 标记用于生产部署的模型
- 📊 **性能对比**: 不同版本模型的性能比较
- 🔄 **版本控制**: 完整的模型生命周期管理
- 👥 **团队协作**: 模型共享和审核工作流

### 3. 训练监控
- 📈 **实时指标**: 训练损失、奖励、成功率等
- 🎯 **模型性能**: 与训练指标关联的模型版本
- 📸 **媒体日志**: 训练过程中的图像、视频记录

## 🛠️ 命令行使用

### 训练命令

```bash
# 基础训练
python train_ppo_with_wandb.py

# 指定配置文件
python train_ppo_with_wandb.py --config-path cfg --config-name wandb_model_config

# 从 wandb artifact 恢复
python train_ppo_with_wandb.py resume_checkpoint="user/navrl-models/model:best"

# 离线模式训练
python train_ppo_with_wandb.py wandb.mode=offline

# 禁用 wandb
python train_ppo_with_wandb.py wandb.mode=disabled
```

### 模型下载和评估

```bash
# 下载模型进行评估
python -c "
from wandb_model_utils import download_model_from_wandb
model_path = download_model_from_wandb('user/project/model:best')
print(f'Model downloaded to: {model_path}')
"
```

## 📋 最佳实践

### 1. 模型版本命名

```python
# 推荐的别名使用
"latest"    # 最新模型
"best"      # 最佳性能模型
"stable"    # 稳定版本
"v1.0"      # 版本标记
"prod"      # 生产版本
```

### 2. 性能阈值设置

```yaml
model_management:
  upload_conditions:
    min_reward_threshold: 50.0      # 只上传奖励 > 50 的模型
    min_success_rate: 0.8           # 只上传成功率 > 80% 的模型
```

### 3. 存储优化

```yaml
model_management:
  upload_frequency: 10              # 减少上传频率节省带宽
  keep_last_n: 3                    # 只保留最近3个版本
```

## 🔍 故障排除

### 常见问题

1. **wandb 上传失败**
   ```python
   # 解决方案：检查网络连接和 API key
   wandb login  # 重新登录
   ```

2. **模型文件过大**
   ```python
   # 解决方案：启用模型压缩
   torch.save(state_dict, path, _use_new_zipfile_serialization=False)
   ```

3. **artifact 下载慢**
   ```python
   # 解决方案：使用本地缓存
   artifact.download(root="./cache")
   ```

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查 wandb 连接状态
print(f"wandb run: {wandb.run.name if wandb.run else 'No active run'}")
```

## 📈 性能监控

### 关键指标跟踪

```python
# 自动跟踪的指标
wandb.log({
    "model/total_parameters": total_params,
    "model/file_size_mb": file_size,
    "train/step": step,
    "eval/mean_reward": reward,
    "best_model/step": best_step
})
```

### 自定义指标

```python
# 添加自定义指标
wandb.log({
    "custom/exploration_rate": exploration_rate,
    "custom/action_diversity": action_entropy,
    "custom/memory_usage": memory_mb
})
```

## 🎉 总结

NavRL 的 wandb 集成为你提供了：

✅ **完整的模型生命周期管理**  
✅ **自动化的版本控制和元数据跟踪**  
✅ **智能的性能基准模型保存**  
✅ **便捷的模型共享和协作**  
✅ **生产级的模型部署支持**  

通过这些功能，你可以：
- 📊 轻松跟踪和比较不同训练运行的模型性能
- 🔄 快速回滚到之前的最佳模型版本  
- 👥 与团队成员分享训练好的模型
- 🚀 将最佳模型部署到生产环境
- 📈 建立完整的模型性能基准数据库

开始使用 NavRL + wandb 模型管理，让你的 RL 训练更加专业和高效！