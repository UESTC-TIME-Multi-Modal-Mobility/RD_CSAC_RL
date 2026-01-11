'''
Author: zdytim zdytim@foxmail.com
Date: 2025-12-18 00:34:11
LastEditors: zdytim zdytim@foxmail.com
LastEditTime: 2026-01-07 00:00:57
FilePath: /u20/NavRL/isaac-training/training/scripts/VIT.py
Description: ViT Encoder for NavRL - 仅包含 encoder_blocks 和 decoder，用于特征提取
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from ViTsubmodules import *
# ==============================================================================
# ViT Feature Extractor (仅用于特征提取，不包含 LSTM/FC2)
# ==============================================================================

class VIT(nn.Module):
    """
    ViT Encoder for Navigation RL - 支持动态输入尺寸
    - Input: [Batch, 1, H, W] (单通道灰度深度图，支持任意尺寸)
    - Output: [Batch, 512] (特征向量)
    - 遵循NavRL编码模式，使用动态尺寸计算
    """
    def __init__(self, input_size=(224, 224)):
        super().__init__()
        self.input_size = input_size
        
        # ViT Encoder: 两层 MixTransformer - 通过大stride控制特征图尺寸
        self.encoder_blocks = nn.ModuleList([
            # 第1层: 224x224 -> 28x28 (stride=8) - 大幅下采样
            MixTransformerEncoderLayer(
                1, 32, 
                patch_size=7, stride=8, padding=3, 
                n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8
            ),
            # 第2层: 28x28 -> 14x14 (stride=2) - 进一步压缩
            MixTransformerEncoderLayer(
                32, 64, 
                patch_size=3, stride=2, padding=1, 
                n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8
            )
        ])

        # 动态计算融合层尺寸
        self._init_dynamic_layers()
        
        print(f"✅ VIT initialized for input size: {input_size}")

    def _init_dynamic_layers(self):
        """根据输入尺寸动态初始化融合层 - 大stride版本"""
        # 计算两层encoder输出尺寸 (使用大stride)
        h1, w1 = self._calc_conv_output_size(self.input_size, 7, 8, 3)  # 第一层: 224->28 (stride=8)
        h2, w2 = self._calc_conv_output_size((h1, w1), 3, 2, 1)        # 第二层: 28->14 (stride=2)
        
        # 融合层配置 - 基于两层输出，大幅减少尺寸
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        # pxShuffle后: [B, 16, h2*2, w2*2] = [B, 16, 28, 28]
        
        # up_sample目标尺寸匹配pxShuffle输出  
        self.target_size = (h2 * 2, w2 * 2)  # (28, 28)
        
        # 融合两层特征: 32 + 16 = 48 channels
        self.down_sample = nn.Conv2d(48, 12, 3, padding=1)
        
        # decoder输入维度大幅减少: 12 * 28 * 28 = 9408 (vs 37632)
        decoder_input_dim = 12 * self.target_size[0] * self.target_size[1]
        self.decoder = spectral_norm(nn.Linear(decoder_input_dim, 512))
        
        print(f"   📐 大Stride版本尺寸计算:")
        print(f"      Layer1 output: {h1}x{w1} (32 channels, stride=8)")
        print(f"      Layer2 output: {h2}x{w2} (64 channels, stride=2)")  
        print(f"      Fusion target: {self.target_size}")
        print(f"      Decoder input: {decoder_input_dim:,} (减少 {((37632-decoder_input_dim)/37632)*100:.1f}%)")
        print(f"      Decoder params: {decoder_input_dim * 512:,} (vs 原始19.3M)")
        print(f"   🎯 保持两层结构，仅通过大stride实现参数优化")

    @staticmethod
    def _calc_conv_output_size(input_size, kernel_size, stride, padding):
        """计算卷积层输出尺寸 - 遵循NavRL utils模式"""
        h, w = input_size
        h_out = (h + 2 * padding - kernel_size) // stride + 1
        w_out = (w + 2 * padding - kernel_size) // stride + 1
        return h_out, w_out

    def forward(self, x):
        """
        前向传播 - 大stride版本，参数优化
        Args:
            x: Tensor [B, 1, H, W] (灰度深度图)
        Returns:
            out: Tensor [B, 512] (特征向量)
        """
        # 输入验证遵循NavRL模式
        assert x.dim() == 4, f"Expected 4D input [B,C,H,W], got {x.shape}"
        assert x.shape[1] == 1, f"Expected 1 channel (grayscale), got {x.shape[1]} channels"
        
        # 如果输入尺寸与期望不符，插值调整
        if x.shape[-2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=True)

        # 两层ViT编码 - 大stride快速下采样
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        
        out1, out2 = embeds[1], embeds[2]  # [B,32,28,28], [B,64,14,14]
        
        # 清理embeds列表
        del embeds
        
        # 两层特征融合
        # Layer2 pxShuffle: [B,64,14,14] -> [B,16,28,28]
        pxshuf_out = self.pxShuffle(out2)  
        
        # Layer1 直接使用: [B,32,28,28] - 已经是目标尺寸
        upsampled_out1 = F.interpolate(out1, size=self.target_size, mode='bilinear', align_corners=True)
        
        # 拼接两层特征: 32 + 16 = 48 channels
        out = torch.cat([upsampled_out1, pxshuf_out], dim=1)  # [B,48,28,28]
        
        # 清理中间变量
        del out1, out2, pxshuf_out, upsampled_out1
        
        # 降维到12通道
        out = self.down_sample(out)  # [B,12,28,28]
        
        # 展平并通过decoder - 参数大幅减少
        out = self.decoder(out.flatten(1))  # [B, 9408] -> [B, 512]
        
        return out
    
class LSTMNetVIT(nn.Module):
    """
    ViT+LSTM Network 
    Num Params: 3,563,663   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = spectral_norm(nn.Linear(4608, 512))
        self.lstm = (nn.LSTM(input_size=517, hidden_size=128,
                         num_layers=3, dropout=0.1))
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))

        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

    def forward(self, X):

        # X = refine_inputs(X)

        x = X[0]
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        if len(X)>3:
            out,h = self.lstm(out, X[3])
        else:
            out,h = self.lstm(out)
        out = self.nn_fc2(out)
        return out, h