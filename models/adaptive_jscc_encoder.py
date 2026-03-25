"""
Adaptive JSCC Encoder
论文: "Timeliness-Aware Joint Source and Channel Coding for Adaptive Image Transmission"

架构: Patch Embedding → 侧信息融合 → 多阶段 Swin Transformer → Masking → Power Norm
输出: 复数域语义码字 x_t ∈ C^K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Patch Embedding
#    将输入图像切块并线性映射为 token 序列
# ============================================================
class PatchEmbedding(nn.Module):
    """
    输入: (B, C, H, W)
    输出: (B, num_patches, embed_dim)
        num_patches = (H/2) * (W/2)
    """
    def __init__(self, img_size=256, patch_size=2, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 用卷积实现: kernel=patch_size, stride=patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)          # (B, embed_dim, H/2, W/2)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x


# ============================================================
# 2. Side Information Embedding
#    将 SNR(γ) 和 CBR(η) 编码为与 patch embedding 同维的向量
# ============================================================
class SideInfoEmbedding(nn.Module):
    """
    输入: snr scalar, cbr scalar  (均为 float tensor, shape (B,))
    输出: (B, num_patches, embed_dim)  — 广播到每个 patch
    """
    def __init__(self, embed_dim=128, num_patches=16384):
        super().__init__()
        self.num_patches = num_patches
        # 每个侧信息用一个线性层映射到 embed_dim
        self.snr_proj = nn.Linear(1, embed_dim)
        self.cbr_proj = nn.Linear(1, embed_dim)

    def forward(self, snr, cbr):
        # snr, cbr: (B,) → (B, 1)
        snr = snr.unsqueeze(-1).float()
        cbr = cbr.unsqueeze(-1).float()
        snr_emb = self.snr_proj(snr)   # (B, embed_dim)
        cbr_emb = self.cbr_proj(cbr)   # (B, embed_dim)
        side_emb = snr_emb + cbr_emb   # (B, embed_dim)
        # 复制到所有 patch 位置
        side_emb = side_emb.unsqueeze(1).expand(-1, self.num_patches, -1)
        return side_emb                # (B, num_patches, embed_dim)


# ============================================================
# 3. Window-based Multi-head Self-Attention (W-MSA)
#    在局部窗口内做自注意力，捕捉局部依赖
# ============================================================
class WindowAttention(nn.Module):
    """
    参数:
        dim        : token 维度
        window_size: 窗口大小 (int, 正方形)
        num_heads  : 注意力头数
        shift      : 是否使用 shifted window (SW-MSA)
    """
    def __init__(self, dim, window_size=8, num_heads=4, shift=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift = shift
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # 相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))  # (2, ws, ws)
        coords_flat = grid.flatten(1)                                       # (2, ws^2)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]       # (2, ws^2, ws^2)
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('relative_position_index', relative.sum(-1))  # (ws^2, ws^2)

    def forward(self, x, H, W):
        """
        x: (B, H*W, dim)
        返回: (B, H*W, dim)
        """
        B, N, C = x.shape
        ws = self.window_size
        shift = ws // 2 if self.shift else 0

        # 重塑为 2D feature map
        x2d = x.view(B, H, W, C)

        # Cyclic shift (SW-MSA)
        if self.shift:
            x2d = torch.roll(x2d, shifts=(-shift, -shift), dims=(1, 2))

        # 分窗口: (B, nH, ws, nW, ws, C) → (B*nW, ws*ws, C)
        nH, nW = H // ws, W // ws
        x2d = x2d.view(B, nH, ws, nW, ws, C)
        x2d = x2d.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_win = x2d.view(B * nH * nW, ws * ws, C)

        # QKV
        Bw, Nw, _ = x_win.shape
        qkv = self.qkv(x_win).reshape(Bw, Nw, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (Bw, heads, Nw, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 相对位置偏置
        pos_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(ws * ws, ws * ws, -1).permute(2, 0, 1)  # (heads, ws^2, ws^2)
        attn = attn + pos_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(Bw, Nw, C)
        out = self.proj(out)

        # 反分窗口
        out = out.view(B, nH, nW, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        # 反 cyclic shift
        if self.shift:
            out = torch.roll(out, shifts=(shift, shift), dims=(1, 2))

        return out.view(B, H * W, C)


# ============================================================
# 4. Swin Transformer Block (一对: W-MSA + SW-MSA)
# ============================================================
class SwinTransformerBlock(nn.Module):
    """
    标准 Swin Block:
        LayerNorm → (W-MSA 或 SW-MSA) → 残差
        LayerNorm → MLP → 残差
    """
    def __init__(self, dim, num_heads=4, window_size=8, shift=False,
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, shift)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, H, W):
        # x: (B, H*W, dim)
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


def make_swin_pair(dim, num_heads, window_size):
    """返回连续两个 Swin Block: W-MSA + SW-MSA"""
    return nn.ModuleList([
        SwinTransformerBlock(dim, num_heads, window_size, shift=False),
        SwinTransformerBlock(dim, num_heads, window_size, shift=True),
    ])


# ============================================================
# 5. Patch Merging
#    空间分辨率 /2，通道数 *2
# ============================================================
class PatchMerging(nn.Module):
    """
    输入: (B, H*W, C)
    输出: (B, (H/2)*(W/2), 2C)
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        # 取 2x2 相邻 patch
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, (H // 2) * (W // 2), 4 * C)
        x = self.norm(x)
        x = self.reduction(x)                     # (B, H/2*W/2, 2C)
        return x, H // 2, W // 2


# ============================================================
# 6. Masking + Power Normalization
#    根据 CBR 只保留前 K 个 token，然后归一化功率
# ============================================================
class MaskingAndPowerNorm(nn.Module):
    """
    - masking   : 根据 cbr 决定保留多少 token (前 K 个)
    - power norm: 将所有 token 的功率归一化到 1
    - 复数化    : 将实数 token 后半维度作为虚部，输出复数码字
    """
    def __init__(self, max_tokens, token_dim):
        super().__init__()
        self.max_tokens = max_tokens
        self.token_dim = token_dim  # 最终 token 维度，须为偶数
        # 线性层将 token 映射到目标维度（实部+虚部）
        self.linear = nn.Linear(token_dim, token_dim)

    def forward(self, x, cbr):
        """
        x   : (B, num_tokens, token_dim)
        cbr : (B,) float in (0, 1]
        返回:
            codeword: (B, K) 复数码字, K = round(cbr * max_tokens)
        """
        B, N, D = x.shape
        x = self.linear(x)  # (B, N, D)

        # --- Masking ---
        # K 个 token，确保 K*D 为偶数（保证复数化无歧义）
        K = max(1, round(cbr[0].item() * self.max_tokens))
        K = min(K, N)
        if (K * D) % 2 != 0:   # K*D 为奇数时多取一个 token
            K = min(K + 1, N)
        x_masked = x[:, :K, :]   # (B, K, D)

        # --- 将 token 展平: (B, K*D) ---
        x_flat = x_masked.reshape(B, -1)   # (B, K*D)，K*D 保证为偶数

        # --- Power Normalization ---
        pwr = x_flat.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        x_norm = x_flat / pwr              # (B, K*D)

        # --- 复数化: 前半为实部，后半为虚部 ---
        half = x_norm.shape[-1] // 2      # 精确等于 K*D//2
        real = x_norm[:, :half]
        imag = x_norm[:, half:]
        codeword = torch.complex(real, imag)  # (B, K*D//2)

        return codeword


# ============================================================
# 7. 完整 Adaptive JSCC Encoder
# ============================================================
class AdaptiveJSCCEncoder(nn.Module):
    """
    完整编码器，对应论文 Fig. 2 左半部分

    参数:
        img_size    : 输入图像边长 (默认 256)
        patch_size  : patch 切分大小 (默认 2 → num_patches = 128*128)
        in_channels : 图像通道数 (默认 3)
        embed_dim   : 初始 embedding 维度 (默认 128)
        num_heads   : 各阶段注意力头数
        window_size : Swin 窗口大小
        depths      : 各阶段 Swin Block 对数 [N1, N2, ..., NM]
    """
    def __init__(
        self,
        img_size=256,
        patch_size=2,
        in_channels=3,
        embed_dim=128,
        num_heads=4,
        window_size=8,
        depths=(3, 3, 3),   # M=3 个阶段，每阶段 3 对 Swin Block
    ):
        super().__init__()

        # Step 1: Patch Embedding
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Step 2: Side Information Embedding (SNR + CBR)
        self.side_embed = SideInfoEmbedding(embed_dim, num_patches)

        # Step 3: 多阶段 Swin Transformer + Patch Merging
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        dim = embed_dim
        for i, depth in enumerate(depths):
            # 每阶段含 depth 对 (W-MSA + SW-MSA) Swin Block
            blocks = nn.ModuleList()
            for _ in range(depth):
                pair = make_swin_pair(dim, num_heads, window_size)
                blocks.append(pair)
            self.stages.append(blocks)
            # 除最后一阶段外加 Patch Merging
            if i < len(depths) - 1:
                self.merges.append(PatchMerging(dim))
                dim = dim * 2  # 通道加倍
            else:
                self.merges.append(None)

        self.final_dim = dim

        # Step 4: 最终线性映射 + Masking + Power Norm
        # max_tokens = 最后阶段的 patch 数
        scale_factor = 2 ** (len(depths) - 1)       # Merging 次数
        h_final = (img_size // patch_size) // scale_factor
        self.H_final = h_final
        self.W_final = h_final
        max_tokens = h_final * h_final

        self.mask_power = MaskingAndPowerNorm(max_tokens, self.final_dim)

    def forward(self, x, snr, cbr):
        """
        参数:
            x   : 原始图像 (B, C, H, W)
            snr : 信道 SNR  (B,)  单位 dB，已归一化
            cbr : 目标 CBR  (B,)  ∈ (0,1]
        返回:
            codeword: 复数语义码字 (B, K_complex)
        """
        B = x.shape[0]

        # --- Patch Embedding ---
        tokens = self.patch_embed(x)   # (B, num_patches, embed_dim)

        # --- 融合 Side Information ---
        side = self.side_embed(snr, cbr)  # (B, num_patches, embed_dim)
        tokens = tokens + side            # 按元素相加融合

        # --- 多阶段 Swin Transformer ---
        H = self.patch_embed.img_size // self.patch_embed.patch_size
        W = H

        for i, (blocks, merge) in enumerate(zip(self.stages, self.merges)):
            for pair in blocks:
                # pair 是 [W-MSA block, SW-MSA block]
                tokens = pair[0](tokens, H, W)  # W-MSA
                tokens = pair[1](tokens, H, W)  # SW-MSA

            if merge is not None:
                tokens, H, W = merge(tokens, H, W)

        # tokens: (B, H_final*W_final, final_dim)

        # --- Masking + Power Normalization → 复数码字 ---
        codeword = self.mask_power(tokens, cbr)  # (B, K_complex)

        return codeword


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    B = 2
    img = torch.randn(B, 3, 256, 256)
    snr = torch.tensor([7.0, 5.0])    # dB
    cbr = torch.tensor([0.06, 0.06])  # 保留 6% tokens

    encoder = AdaptiveJSCCEncoder(
        img_size=256,
        patch_size=2,
        in_channels=3,
        embed_dim=128,
        num_heads=4,
        window_size=8,
        depths=(3, 3, 3),
    )

    with torch.no_grad():
        codeword = encoder(img, snr, cbr)

    print("=" * 50)
    print(f"输入图像形状  : {list(img.shape)}")
    print(f"SNR (dB)     : {snr.tolist()}")
    print(f"CBR          : {cbr.tolist()}")
    print(f"输出码字形状  : {list(codeword.shape)}  (复数)")
    print(f"码字数据类型  : {codeword.dtype}")
    real_pwr = (codeword.real.pow(2) + codeword.imag.pow(2)).mean()
    print(f"平均发射功率  : {real_pwr.item():.4f}  (归一化后应≈1)")
    print("=" * 50)

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"编码器总参数量: {total_params:,}")
