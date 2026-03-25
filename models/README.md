# Adaptive JSCC Encoder — 验证指南

## 文件结构

```
jscc/
├── adaptive_jscc_encoder.py   # Encoder（论文 Fig.2 左半部分）
├── adaptive_jscc_decoder.py   # Decoder（镜像结构）
├── awgn_channel.py            # AWGN 信道模拟（论文公式 2）
├── validate.py                # 完整验证脚本
└── README.md
```

---

## 环境安装

```bash
pip install torch torchvision matplotlib pillow numpy
```

---

## 运行验证

```bash
cd jscc
python validate.py
```

运行结束后会生成：
- `validation_results.png` — 四合一图表（见下）
- `jscc_checkpoint.pth`   — 训练好的模型权重

---

## 验证内容说明

### (a) 训练曲线
观察 MSE Loss 下降 + PSNR 上升，确认模型收敛。

### (b) PSNR vs CBR（对应论文 Fig.5）
固定 SNR = 5 dB，扫描 CBR ∈ {0.02, 0.04, ..., 0.12}
- **期望结果**：CBR 越大，PSNR 越高
- **对比 JPEG Baseline**：JSCC 在低 CBR 下应明显优于 JPEG

### (c) PSNR vs SNR（对应论文 Fig.6）
固定 CBR = 0.06，扫描 SNR ∈ {3, 5, 7, 9, 11} dB
- **期望结果**：SNR 越高，PSNR 越高
- JPEG 没有信道，作为水平基准线

### (d) 重建图像可视化
3 张测试图，上行原图，下行重建图
- **期望结果**：主体结构可辨，细节随 CBR/SNR 有所损失

---

## 调整配置

在 `validate.py` 的 `CFG` 字典中修改：

| 参数 | 说明 | 默认值 | 论文设置 |
|------|------|--------|---------|
| `img_size` | 图像边长 | 128 | 256 |
| `embed_dim` | 初始 embedding 维度 | 64 | 128 |
| `depths` | 各阶段 Block 对数 | (2,2) | (3,3,3) |
| `epochs` | 训练轮数 | 30 | 200 |
| `batch_size` | Batch 大小 | 4 | 8 |

> 将配置改为论文原始值（img_size=256, depths=(3,3,3), epochs=200）
> 可复现论文中的 PSNR 数值，但需要更长训练时间和更多显存。

---

## 用真实数据集替换

将 `validate.py` 中的 `RandomImageDataset` 替换为：

```python
from torchvision.datasets import ImageFolder

train_set = ImageFolder('path/to/buoycam/train', transform=T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
]))
```

---

## 已知限制

- 当前验证使用随机生成图像（含高斯模糊），
  PSNR 绝对值会低于论文（论文使用真实海洋图像 BuoyCAM）
- 验证脚本未包含 Stage 2（PPO 码长分配），
  仅验证 JSCC 编解码质量本身
