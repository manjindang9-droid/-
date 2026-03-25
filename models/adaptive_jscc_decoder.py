"""
Adaptive JSCC Decoder
对应论文 Fig. 2 右半部分，镜像 Encoder 结构：
Zero-Padding → Linear Embedding → 多阶段 Swin Transformer → Patch Division → Conv2d 重建
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_jscc_encoder import SwinTransformerBlock, make_swin_pair


# ============================================================
# Patch Expansion (PatchMerging 的逆操作)
#   空间分辨率 *2，通道数 /2
# ============================================================
class PatchExpansion(nn.Module):
    """
    输入: (B, H*W, C)
    输出: (B, (H*2)*(W*2), C//2)
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = self.norm(x)
        x = self.expand(x)           # (B, H*W, 2C)
        x = x.view(B, H, W, 2 * C)
        # 将通道拆成 2x2 空间 patch
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = x.view(B, H * 2 * W * 2, C // 2)
        return x, H * 2, W * 2


# ============================================================
# 完整 Adaptive JSCC Decoder
# ============================================================
class AdaptiveJSCCDecoder(nn.Module):
    """
    参数须与 Encoder 对称：
        final_dim   : Encoder 最后阶段输出的通道数
        H_final     : Encoder 最后阶段的空间边长
        depths      : 各阶段 Swin Block 对数（与 Encoder 相同顺序，解码时倒序展开）
        img_size    : 重建目标图像边长
        out_channels: 重建图像通道数
    """
    def __init__(
        self,
        final_dim=512,
        H_final=32,
        depths=(3, 3, 3),
        img_size=256,
        patch_size=2,
        out_channels=3,
        num_heads=4,
        window_size=8,
    ):
        super().__init__()
        self.H_final = H_final
        self.W_final = H_final
        self.final_dim = final_dim

        # Step 1: 线性层将复数码字展平后映射回 token 序列
        # 复数 codeword 展开: real + imag → 实数，再 reshape 为 token
        self.input_proj = nn.Linear(final_dim, final_dim)

        # Step 2: 多阶段 Swin Transformer + Patch Expansion（倒序）
        # depths 对应 Encoder 各阶段，解码时从深层到浅层展开
        self.stages = nn.ModuleList()
        self.expansions = nn.ModuleList()

        dim = final_dim
        for i, depth in enumerate(reversed(depths)):
            blocks = nn.ModuleList()
            for _ in range(depth):
                pair = make_swin_pair(dim, num_heads, window_size)
                blocks.append(pair)
            self.stages.append(blocks)
            if i < len(depths) - 1:
                self.expansions.append(PatchExpansion(dim))
                dim = dim // 2
            else:
                self.expansions.append(None)

        self.final_dim_out = dim  # 最浅层输出通道

        # Step 3: 从 token 重建为像素图像
        # token: (B, num_patches, final_dim_out) → 图像 (B, C, H, W)
        # 用 ConvTranspose2d 做 patch 反投影
        self.norm = nn.LayerNorm(self.final_dim_out)
        self.patch_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.final_dim_out, 64,
                               kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),   # 输出像素值归一化到 [0,1]
        )

    def forward(self, codeword, cbr, max_tokens):
        """
        参数:
            codeword  : 接收到的（含噪）复数码字 (B, K_complex)
            cbr       : 当前 CBR (B,) — 用于推算保留的 token 数
            max_tokens: Encoder 最终阶段 token 总数 (H_final * W_final)
        返回:
            x_hat: 重建图像 (B, C, H, W) ∈ [0,1]
        """
        B = codeword.shape[0]
        D = self.final_dim
        H, W = self.H_final, self.W_final

        # --- 复数→实数展开 ---
        real = codeword.real   # (B, K_complex)
        imag = codeword.imag
        x_flat = torch.cat([real, imag], dim=-1)   # (B, K_real = 2*K_complex)

        # --- Zero-Padding 补回完整 token 序列 ---
        # full_len 必须是 D 的整数倍，且与 Encoder 保证的偶数对齐
        full_len = max_tokens * D
        current_len = x_flat.shape[-1]
        pad_len = full_len - current_len
        if pad_len < 0:
            # 极少情况：CBR=1.0 时可能超出，截断到整数 token 边界
            x_flat = x_flat[:, :full_len]
            pad_len = 0
        if pad_len > 0:
            x_flat = F.pad(x_flat, (0, pad_len))

        # reshape → (B, max_tokens, D)
        tokens = x_flat.view(B, max_tokens, D)
        tokens = self.input_proj(tokens)   # 线性映射

        # --- 多阶段 Swin Transformer + Patch Expansion ---
        for i, (blocks, expansion) in enumerate(zip(self.stages, self.expansions)):
            for pair in blocks:
                tokens = pair[0](tokens, H, W)   # W-MSA
                tokens = pair[1](tokens, H, W)   # SW-MSA
            if expansion is not None:
                tokens, H, W = expansion(tokens, H, W)

        # --- token → 像素图像 ---
        tokens = self.norm(tokens)                       # (B, H*W, C)
        _, C = tokens.shape[1], tokens.shape[2]
        feat = tokens.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_hat = self.patch_upsample(feat)                # (B, out_channels, img_size, img_size)

        return x_hat
