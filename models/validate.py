"""
完整验证脚本
验证内容:
  1. 重建质量 (PSNR)     — 训练并评估端到端重建效果
  2. 自适应性            — 扫描不同 CBR / SNR，画出曲线（对应论文 Fig.5 / Fig.6）
  3. 与 Baseline 对比   — 对比 JPEG 传统编码

运行:
  python validate.py
"""

import sys, os
import shutil
import datetime as dt

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../models
_ROOT_DIR   = os.path.join(_MODELS_DIR, '..')                     # 项目根目录
_CKPT_DIR   = os.path.join(_ROOT_DIR, 'checkpoints')              # .../checkpoints

sys.path.insert(0, _MODELS_DIR)
# Ensure project root is importable (for `checkpoint_utils.py`)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from checkpoint_utils import make_run_id, write_json, ensure_dir

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import io, math
from PIL import Image

from adaptive_jscc_encoder import AdaptiveJSCCEncoder
from adaptive_jscc_decoder import AdaptiveJSCCDecoder
from awgn_channel       import AWGNChannel


# ============================================================
# 0. 配置
# ============================================================
CFG = dict(
    img_size    = 128,      # 验证用小尺寸，加快训练；论文用 256
    patch_size  = 2,
    in_channels = 3,
    embed_dim   = 64,       # 轻量化，验证用；论文用 128
    num_heads   = 4,
    window_size = 4,        # img_size=128 → feature map=64, ws=4 合适
    depths      = (2, 2),   # 2 个阶段；论文用 3 阶段
    # 训练
    lr          = 1e-4,
    batch_size  = 4,
    epochs      = 100,   # 建议 ≥100 使 PSNR 达到 d_min=28dB；论文用 200
    # 信道
    train_snr_range = (1, 15),   # dB，训练时随机采样
    train_cbr_list  = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
    # 评估
    eval_snr_list   = [3, 5, 7, 9, 11],
    eval_cbr_list   = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
    fixed_eval_snr  = 5,    # 固定 SNR 扫 CBR (Fig.5)
    fixed_eval_cbr  = 0.06, # 固定 CBR 扫 SNR (Fig.6)
    device      = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_test_imgs = 200,    # 随机生成的测试图数
    seed        = 42,
)
print(f"Device: {CFG['device']}")


# ============================================================
# 1. 随机图像数据集（替代 BuoyCAM）
# ============================================================
class RandomImageDataset(Dataset):
    """
    生成随机 RGB 图像（模拟自然图像分布的简单替代）
    实际使用时替换为 ImageFolder 或自定义 Dataset
    """
    def __init__(self, n=1000, img_size=128, seed=42):
        torch.manual_seed(seed)
        self.data = torch.rand(n, 3, img_size, img_size)
        # 加入低频结构使其更接近自然图像
        blur = T.GaussianBlur(kernel_size=15, sigma=5.0)
        self.data = torch.stack([blur(img) for img in self.data])
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]


# ============================================================
# 2. JSCC 完整系统（Encoder + Channel + Decoder）
# ============================================================
class JSCCSystem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = AdaptiveJSCCEncoder(
            img_size    = cfg['img_size'],
            patch_size  = cfg['patch_size'],
            in_channels = cfg['in_channels'],
            embed_dim   = cfg['embed_dim'],
            num_heads   = cfg['num_heads'],
            window_size = cfg['window_size'],
            depths      = cfg['depths'],
        )
        # 推算 decoder 参数
        scale = 2 ** (len(cfg['depths']) - 1)
        h_final = (cfg['img_size'] // cfg['patch_size']) // scale
        final_dim = cfg['embed_dim'] * scale

        self.decoder = AdaptiveJSCCDecoder(
            final_dim   = final_dim,
            H_final     = h_final,
            depths      = cfg['depths'],
            img_size    = cfg['img_size'],
            patch_size  = cfg['patch_size'],
            out_channels= cfg['in_channels'],
            num_heads   = cfg['num_heads'],
            window_size = cfg['window_size'],
        )
        self.channel   = AWGNChannel()
        self.max_tokens = h_final * h_final

    def forward(self, x, snr, cbr):
        codeword = self.encoder(x, snr, cbr)
        noisy    = self.channel(codeword, snr)
        x_hat    = self.decoder(noisy, cbr, self.max_tokens)
        return x_hat


# ============================================================
# 3. PSNR 计算
# ============================================================
def compute_psnr(original, reconstructed):
    """
    original, reconstructed: (B, C, H, W) ∈ [0,1]
    返回: 平均 PSNR (dB)
    """
    mse = F.mse_loss(reconstructed, original, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])           # (B,)
    psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
    return psnr.mean().item()

import torch.nn.functional as F


# ============================================================
# 4. JPEG Baseline（传统编码对比）
# ============================================================
def jpeg_psnr(images, quality):
    """
    对一批图像做 JPEG 压缩，返回平均 PSNR
    quality: JPEG 质量参数 (1~95)
    """
    psnrs = []
    for img in images:
        # Tensor → PIL
        pil = T.ToPILImage()(img.cpu())
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        recon = T.ToTensor()(Image.open(buf))
        mse = F.mse_loss(recon, img.cpu()).item()
        if mse < 1e-10:
            psnrs.append(100.0)
        else:
            psnrs.append(10 * math.log10(1.0 / mse))
    return float(np.mean(psnrs))

def cbr_to_jpeg_quality(cbr, img_size=128):
    """
    粗略将 CBR 映射到 JPEG quality 参数
    CBR = compressed_bits / (C*H*W*bits_per_pixel)
    """
    # 近似：quality ≈ cbr * 100 * 8（非线性，简化处理）
    q = int(cbr * 600)
    return max(1, min(95, q))


# ============================================================
# 5. 训练
# ============================================================
def train(cfg):
    device = cfg['device']
    dataset = RandomImageDataset(n=800, img_size=cfg['img_size'], seed=cfg['seed'])
    loader  = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)

    model = JSCCSystem(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    print("\n" + "="*55)
    print("  Stage 1: Training Adaptive JSCC")
    print("="*55)
    print(f"  {'Epoch':>6} | {'Loss':>10} | {'PSNR (dB)':>10}")
    print("-"*55)

    loss_history, psnr_history = [], []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        epoch_loss, epoch_psnr = 0.0, 0.0

        for batch in loader:
            x = batch.to(device)
            B = x.shape[0]

            # 随机采样 SNR 和 CBR
            snr = torch.FloatTensor(B).uniform_(*cfg['train_snr_range']).to(device)
            cbr_val = np.random.choice(cfg['train_cbr_list'])
            cbr = torch.full((B,), cbr_val, device=device)

            x_hat = model(x, snr, cbr)

            # MSE Loss（对应论文公式 7）
            loss = F.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                p = compute_psnr(x, x_hat)
            epoch_loss += loss.item()
            epoch_psnr += p

        scheduler.step()
        n_batch = len(loader)
        avg_loss = epoch_loss / n_batch
        avg_psnr = epoch_psnr / n_batch
        loss_history.append(avg_loss)
        psnr_history.append(avg_psnr)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6} | {avg_loss:>10.5f} | {avg_psnr:>9.2f}dB")

    print("="*55)
    return model, loss_history, psnr_history


# ============================================================
# 6. 评估函数
# ============================================================
@torch.no_grad()
def eval_psnr(model, dataset, snr_val, cbr_val, device, n=200):
    model.eval()
    indices = torch.randperm(len(dataset))[:n]
    psnrs = []
    bs = 8
    for start in range(0, n, bs):
        idx = indices[start:start+bs]
        x   = torch.stack([dataset[i] for i in idx]).to(device)
        B   = x.shape[0]
        snr = torch.full((B,), snr_val, device=device)
        cbr = torch.full((B,), cbr_val, device=device)
        x_hat = model(x, snr, cbr)
        psnrs.append(compute_psnr(x, x_hat))
    return float(np.mean(psnrs))


# ============================================================
# 7. 绘图
# ============================================================
def plot_results(cfg, model, dataset, loss_hist, psnr_hist):
    device = cfg['device']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Adaptive JSCC Encoder Validation', fontsize=15, fontweight='bold')

    # --- (a) 训练曲线 ---
    ax = axes[0, 0]
    epochs = range(1, len(loss_hist) + 1)
    ax2 = ax.twinx()
    ax.plot(epochs, loss_hist, 'b-o', ms=3, label='MSE Loss')
    ax2.plot(epochs, psnr_hist, 'r-s', ms=3, label='PSNR')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss', color='b')
    ax2.set_ylabel('PSNR (dB)', color='r')
    ax.set_title('(a) Training Curve')
    ax.grid(True, alpha=0.3)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc='upper right', fontsize=8)

    # --- (b) Fig.5: PSNR vs CBR（固定 SNR）---
    ax = axes[0, 1]
    cbrs = cfg['eval_cbr_list']
    fixed_snr = cfg['fixed_eval_snr']

    jscc_psnrs  = [eval_psnr(model, dataset, fixed_snr, c, device) for c in cbrs]
    jpeg_psnrs  = [jpeg_psnr(
                    torch.stack([dataset[i] for i in torch.randperm(len(dataset))[:50]]),
                    cbr_to_jpeg_quality(c, cfg['img_size'])
                  ) for c in cbrs]

    ax.plot(cbrs, jscc_psnrs, 'b-o', label=f'Adaptive JSCC (Ours)')
    ax.plot(cbrs, jpeg_psnrs, 'k--^', label='JPEG (Baseline)')
    ax.set_xlabel('Channel Bandwidth Ratio (CBR) η')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'(b) PSNR vs CBR  [SNR = {fixed_snr} dB]')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- (c) Fig.6: PSNR vs SNR（固定 CBR）---
    ax = axes[1, 0]
    snrs = cfg['eval_snr_list']
    fixed_cbr = cfg['fixed_eval_cbr']

    jscc_by_snr = [eval_psnr(model, dataset, s, fixed_cbr, device) for s in snrs]
    # JPEG 不受 SNR 影响（无信道），作为水平基准
    jpeg_base   = jpeg_psnr(
                    torch.stack([dataset[i] for i in torch.randperm(len(dataset))[:50]]),
                    cbr_to_jpeg_quality(fixed_cbr, cfg['img_size'])
                  )

    ax.plot(snrs, jscc_by_snr, 'b-o', label='Adaptive JSCC (Ours)')
    ax.axhline(jpeg_base, color='k', linestyle='--', label='JPEG (no channel)')
    ax.set_xlabel('SNR (dB) γ')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'(c) PSNR vs SNR  [CBR = {fixed_cbr}]')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- (d) 图像重建可视化 ---
    ax = axes[1, 1]
    model.eval()
    idx = torch.randperm(len(dataset))[:3]
    samples = torch.stack([dataset[i] for i in idx]).to(device)
    snr_vis = torch.full((3,), 7.0, device=device)
    cbr_vis = torch.full((3,), 0.06, device=device)
    with torch.no_grad():
        recon = model(samples, snr_vis, cbr_vis)

    # 拼接：原图 | 重建图
    compare = torch.cat([samples.cpu(), recon.cpu()], dim=0)  # (6, C, H, W)
    grid = compare.permute(0, 2, 3, 1).numpy()  # (6, H, W, C)
    combined = np.concatenate([
        np.concatenate(grid[:3], axis=1),   # 原图横排
        np.concatenate(grid[3:], axis=1),   # 重建横排
    ], axis=0)
    ax.imshow(combined.clip(0, 1))
    ax.set_title('(d) Original (top) vs Reconstructed (bottom)\n[SNR=7dB, CBR=0.06]')
    ax.axis('off')

    plt.tight_layout()
    out_path = 'validation_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  图表已保存: {out_path}")
    plt.close()
    return out_path


# ============================================================
# 8. 数值汇总报告
# ============================================================
def print_summary(cfg, model, dataset):
    device = cfg['device']
    print("\n" + "="*60)
    print("  评估报告: PSNR (dB) @ 不同 CBR / SNR 组合")
    print("="*60)
    cbrs = cfg['eval_cbr_list']
    snrs = cfg['eval_snr_list']

    # 表头
    col_label = 'CBR/SNR'
    header = f"{col_label:>10}" + "".join(f"  {s:>5}dB" for s in snrs)
    print(header)
    print("-" * len(header))

    for c in cbrs:
        row = f"{c:>10.2f}"
        for s in snrs:
            p = eval_psnr(model, dataset, s, c, device, n=50)
            row += f"  {p:>6.2f}"
        print(row)
    print("="*60)

    # 参数量统计
    enc_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    dec_params = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    print(f"\n  Encoder 参数量: {enc_params:.2f} M")
    print(f"  Decoder 参数量: {dec_params:.2f} M")
    print(f"  合计:           {enc_params + dec_params:.2f} M")


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    torch.manual_seed(CFG['seed'])

    # 1. 数据集
    train_set = RandomImageDataset(n=800,  img_size=CFG['img_size'], seed=CFG['seed'])
    test_set  = RandomImageDataset(n=CFG['num_test_imgs'], img_size=CFG['img_size'], seed=CFG['seed']+1)

    # 2. 训练
    model, loss_hist, psnr_hist = train(CFG)

    # 3. 数值报告
    print_summary(CFG, model, test_set)

    # 4. 可视化
    out_path = plot_results(CFG, model, test_set, loss_hist, psnr_hist)

    # 5. 保存模型（带历史记录 + latest 固定文件）
    os.makedirs(_CKPT_DIR, exist_ok=True)

    depths_str = ''.join(str(d) for d in CFG['depths'])   # (2,2) → '22'
    run_id = make_run_id(
        prefix="stage1",
        cfg=CFG,
        extra={"script": "models/validate.py"},
    )

    stage1_dir = os.path.join(_CKPT_DIR, "stage1")
    history_dir = os.path.join(stage1_dir, "history", run_id)
    latest_dir = os.path.join(stage1_dir, "latest")
    ensure_dir(history_dir)
    ensure_dir(latest_dir)

    ckpt_name = f"jscc_ep{CFG['epochs']}_dim{CFG['embed_dim']}_d{depths_str}_{run_id}.pth"
    ckpt_path = os.path.join(history_dir, ckpt_name)

    ckpt = {
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
        'cfg'    : CFG,
        'epochs_trained': CFG['epochs'],       # 记录训练轮数，方便续训判断
        'final_psnr'    : psnr_hist[-1] if psnr_hist else None,
    }
    ckpt["run_id"] = run_id
    ckpt["depths_str"] = depths_str
    torch.save(ckpt, ckpt_path)

    # latest 固定名（Stage2 默认读取用；同时保留旧的根目录固定名兼容）
    latest_path = os.path.join(latest_dir, "jscc_checkpoint.pth")
    torch.save(ckpt, latest_path)

    default_path = os.path.join(_CKPT_DIR, "jscc_checkpoint.pth")  # 兼容旧脚本
    torch.save(ckpt, default_path)

    # 保存元数据（便于追溯：用同一 run_id 能定位 cfg/psnr/路径）
    meta = {
        "run_id": run_id,
        "script": "models/validate.py",
        "timestamp_utc": dt.datetime.utcnow().isoformat(),
        "cfg": CFG,
        "final_psnr": psnr_hist[-1] if psnr_hist else None,
        "epochs_trained": CFG["epochs"],
        "checkpoint_path": ckpt_path,
        "latest_path": latest_path,
        "default_compat_path": default_path,
        # 可选：训练曲线用于复现实验（长度不大）
        "loss_history": loss_hist,
        "psnr_history": psnr_hist,
    }
    write_json(os.path.join(history_dir, "meta.json"), meta)

    # 同时把验证图拷贝到 history_dir 里
    if os.path.exists(out_path):
        try:
            shutil.copy2(out_path, os.path.join(history_dir, "validation_results.png"))
        except Exception:
            pass

    print(f"\n  模型权重已保存:")
    print(f"    带版本名: {ckpt_path}")
    print(f"    latest: {latest_path}  (推荐读取)")
    print(f"    默认名称(兼容): {default_path}  (Stage 2 旧默认读取)")
    print("\n  验证完成！")
