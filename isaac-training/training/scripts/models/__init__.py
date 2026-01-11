'''
Author: zdytim zdytim@foxmail.com
Date: 2026-01-05 22:20:22
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-05 22:20:23
FilePath: /NavRL/isaac-training/training/scripts/models/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
NavRL Models Package
===================
抽象的模型管理模块

包含：
- navrl_model.py: 主要的模型定义和管理器
- 其他未来扩展的模型文件

作者: NavRL Team
日期: 2026年1月5日
"""

from .navrl_model import (
    SharedFeatureExtractor,
    NavRLModel, 
    ModelManager,
    create_navrl_model,
    load_pretrained_model
)

__all__ = [
    'SharedFeatureExtractor',
    'NavRLModel',
    'ModelManager', 
    'create_navrl_model',
    'load_pretrained_model'
]