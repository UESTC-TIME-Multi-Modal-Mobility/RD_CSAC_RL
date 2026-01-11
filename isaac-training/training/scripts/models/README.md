<!--
 * @Author: zdytim zdytim@foxmail.com
 * @Date: 2026-01-05 22:22:34
 * @LastEditors: zdytim zdytim@foxmail.com
 * @LastEditTime: 2026-01-06 00:00:00
 * @FilePath: /NavRL/isaac-training/training/scripts/models/README.md
 * @Description: NavRL 模型管理模块文档
-->
# NavRL 模型管理器

## 概述

抽象化的模型管理模块，提供统一的模型创建、保存、加载和管理接口。

## 模块结构

### 1. PPO-ViT 模型 (`navrl_model.py`)

用于基于 Vision Transformer 的 PPO 算法。

**主要组件：**
- `SharedFeatureExtractor`: ViT-based 特征提取器
- `NavRLModel`: 完整的 PPO-ViT 模型
- `ModelManager`: PPO 模型管理器

### 2. SAC 模型 (`sac_model.py`) 🆕

用于 Soft Actor-Critic 算法的模型管理。

**主要组件：**
- `SACFeatureExtractor`: CNN-based 特征提取器
- `ActorNetwork`: SAC Actor 网络
- `CriticNetwork`: SAC Critic 网络（Q函数）
- `SACModel`: 完整的 SAC 模型
- `SACModelManager`: SAC 模型管理器

**使用示例：**
```python
from models.sac_model import SACModelManager

# 创建 SAC 模型管理器
manager = SACModelManager(
    cfg=cfg,
    observation_spec=env.observation_spec,
    action_spec=env.action_spec,
    device=device
)

# 推理
with torch.no_grad():
    actions = manager.get_action(observations, deterministic=True)

# 训练步骤
loss_info = manager.train_step(replay_buffer, batch_size=256, tau=0.005)

# 保存/加载检查点
manager.save_checkpoint("sac_checkpoint.pt", step=10000)
manager.load_checkpoint("sac_checkpoint.pt")
```

**在训练脚本中使用：**
```python
# 方式1：直接使用 SACModelManager
from models.sac_model import SACModelManager
sac_agent = SACModelManager(cfg, obs_spec, act_spec, device)

# 方式2：使用 SAC_V2 包装类（向后兼容）
from SAC_v1 import SAC_V2
sac_agent = SAC_V2(cfg, obs_spec, act_spec, device)
```

## 设计原则

1. **关注点分离**：模型定义与训练逻辑分离
2. **统一接口**：所有模型管理器提供一致的 API
3. **易于扩展**：新模型只需继承基类并实现关键方法
4. **向后兼容**：保留原有接口，便于逐步迁移
5. **完整封装**：包含模型、优化器、检查点管理等所有组件

## 迁移指南

### 从旧版 SAC 迁移到 SAC_V2

**旧代码：**
```python
from SAC_v1 import SAC
sac_agent = SAC(cfg, obs_spec, act_spec, device)
```

**新代码（推荐）：**
```python
from SAC_v1 import SAC_V2  # 使用 V2 版本
sac_agent = SAC_V2(cfg, obs_spec, act_spec, device)
```

接口保持一致，无需修改其他代码。

## 📋 概述

NavRL 模型管理器是一个抽象的模型管理系统，将PPO-ViT模型的核心逻辑从训练脚本中分离出来，提供统一的模型创建、加载、保存和配置管理接口。

## 🏗️ 架构设计

```
models/
├── __init__.py              # 包导入定义
├── navrl_model.py          # 主要模型定义和管理器
└── README.md              # 本文档
```

### 核心组件

1. **SharedFeatureExtractor**: ViT-based特征提取器
   - 支持ViT backbone的加载和参数管理
   - 动态障碍物编码器
   - 状态编码器和特征融合网络

2. **NavRLModel**: 完整的PPO-ViT模型
   - Actor/Critic网络头
   - 优化器和训练工具
   - 混合精度训练支持

3. **ModelManager**: 统一的模型管理接口
   - 检查点加载/保存
   - 参数管理（冻结/解冻）
   - 配置和状态管理

## 🚀 快速开始

### 基础使用

```python
from models import create_navrl_model

# 创建新模型
model_manager = create_navrl_model(cfg, obs_spec, act_spec, device)
model = model_manager.get_model()

# 训练循环
for data in collector:
    train_info = model.train(data)
    print(f"Loss: {train_info['total_loss']:.4f}")
```

### 加载预训练模型

```python
from models import load_pretrained_model

# 从检查点加载
model_manager = load_pretrained_model(
    checkpoint_path="checkpoint.pt",
    cfg=cfg,
    observation_spec=obs_spec,
    action_spec=act_spec,
    device=device
)
```

## 🔧 高级功能

### 参数管理

```python
# 查看模型摘要
model_manager.print_model_summary()

# 冻结ViT encoder（仅训练decoder）
model_manager.freeze_vit_encoder()

# 解冻所有ViT参数（完整fine-tuning）
model_manager.unfreeze_all_vit()

# 获取参数统计
info = model.get_model_info()
print(f"Trainable params: {info['trainable_parameters']:,}")
```

### 检查点管理

```python
# 保存检查点
model_manager.save_checkpoint(
    filepath="checkpoint.pt",
    step=1000,
    additional_info={'custom_data': 'value'}
)

# 加载检查点
success = model_manager.load_checkpoint("checkpoint.pt")
```

### 训练模式切换

```python
# 设置训练模式
model_manager.set_training_mode(True)

# 设置评估模式
model_manager.set_training_mode(False)
```

## 📊 模型信息

使用 `model_manager.print_model_summary()` 可以查看详细的模型信息：

```
============================================================
📊 NavRL Model Summary
============================================================
🏗️  Architecture: PPO-ViT with Shared Feature Extractor
🎯 Action Dimension: 3
💾 Device: cuda:0
⚡ Mixed Precision: Enabled
📈 Total Parameters: 2,456,789
🔄 Trainable Parameters: 1,234,567
❄️  Frozen Parameters: 1,222,222
   - ViT Decoder: 987,654
   - Other Modules: 246,913
============================================================
```

## 🔄 从现有代码迁移

### 迁移步骤

1. **导入更改**
```python
# OLD
from ppo_vit_v3 import PPOVIT

# NEW
from models import create_navrl_model, load_pretrained_model
```

2. **模型创建**
```python
# OLD
model = PPOVIT(cfg, observation_spec, action_spec, device)

# NEW
model_manager = create_navrl_model(cfg, observation_spec, action_spec, device)
model = model_manager.get_model()
```

3. **检查点处理**
```python
# OLD
model.load_full_checkpoint(checkpoint_path)
torch.save({'model_state_dict': model.state_dict()}, path)

# NEW
model_manager = load_pretrained_model(checkpoint_path, cfg, obs_spec, act_spec, device)
model_manager.save_checkpoint(path, step=step)
```

### 完整示例

参考 `train_migrated.py` 了解完整的迁移示例。

## 📝 配置选项

### 模型配置

```yaml
feature_extractor:
  pretrained_checkpoint_path: "path/to/vit_weights.pth"  # ViT预训练权重路径

actor:
  learning_rate: 3e-4  # 基础学习率

use_amp: true  # 混合精度训练
```

### 训练配置

```yaml
training:
  mode: "basic"  # 训练模式: basic, advanced, finetune
  freeze_vit_encoder: true  # 是否冻结ViT编码器
  full_vit_finetune: false  # 是否进行完整ViT微调
```

## ✨ 主要优势

1. **模块化设计**: 清晰的责任分离，易于维护和扩展
2. **统一接口**: 简化了模型创建、加载、保存等操作
3. **参数管理**: 灵活的ViT参数冻结/解冻策略
4. **错误处理**: 完善的检查点加载错误处理和统计
5. **性能优化**: 支持混合精度训练和分组学习率
6. **可读性**: 清晰的代码结构和丰富的文档

## 🛠️ 扩展指南

### 添加新的特征提取器

1. 在 `navrl_model.py` 中继承 `SharedFeatureExtractor`
2. 重写 `forward` 方法
3. 更新 `ModelManager` 以支持新的配置

### 添加新的训练策略

1. 在 `NavRLModel` 中添加新的训练方法
2. 更新配置文件以支持新参数
3. 在 `ModelManager` 中添加相应的管理接口

## 📚 更多示例

- `train_with_manager.py`: 完整的训练器实现
- `train_migrated.py`: 从现有代码的迁移示例
- 各个模块的docstring中包含详细的API文档

## 🐛 故障排除

### 常见问题

1. **导入错误**: 确保在正确的目录下运行，并且Python路径包含项目根目录
2. **检查点加载失败**: 检查文件路径和模型架构匹配性
3. **参数冻结不生效**: 确保在优化器创建之前设置参数状态

### 调试技巧

- 使用 `model_manager.print_model_summary()` 查看模型状态
- 检查 `model.get_model_info()` 获取详细参数信息
- 查看检查点加载时的统计输出

## 📞 支持

如有问题或建议，请联系NavRL开发团队或提交Issue。

---

**NavRL Team** - 2026年1月5日