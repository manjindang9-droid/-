"""
AWGN 信道模拟
对应论文公式 (2): y_t' = x_t + n_t'
"""

import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    """
    加性高斯白噪声信道

    SNR (dB) → 噪声功率 → 在复数码字上叠加噪声

    注意：Encoder 已做功率归一化，信号功率 ≈ 1
    因此: noise_var = 1 / (10^(SNR_dB/10))
    """
    def forward(self, codeword, snr_db):
        """
        参数:
            codeword: 复数码字 (B, K)
            snr_db  : 每个样本的 SNR (B,) 单位 dB
        返回:
            noisy_codeword: 含噪复数码字 (B, K)
        """
        # 噪声方差: σ² = 1 / SNR_linear
        snr_linear = 10 ** (snr_db / 10.0)           # (B,)
        noise_var = 1.0 / snr_linear                  # (B,)
        noise_std = (noise_var / 2).sqrt()            # 实部虚部各分 1/2 功率

        # 生成复数噪声
        noise_std = noise_std.to(codeword.device).float()
        noise_real = torch.randn_like(codeword.real) * noise_std.unsqueeze(-1)
        noise_imag = torch.randn_like(codeword.imag) * noise_std.unsqueeze(-1)
        noise = torch.complex(noise_real, noise_imag)

        return codeword + noise
