# PythonProject7

该项目包含三部分内容：

1. `models/`：Stage-1 自适应 JSCC 编码器/解码器与信道（`adaptive_jscc_encoder.py` / `adaptive_jscc_decoder.py` / `awgn_channel.py`）以及 Stage-1 验证脚本（`validate.py`）。
2. `ppo/`：Stage-2 的强化学习环境与训练脚本。
   - PPO：`ppo/train_stage2.py`
   - DiscoRL（Disco103）：`ppo/train_stage2_disco.py`
3. `disco_rl-main/`：DeepMind 官方 `disco_rl` 源码（内部有 `disco_rl/` 包），Stage-2 DiscoRL 依赖。

## 环境配置（关键点）

- 训练脚本会使用相对路径把项目目录加入 `sys.path`，因此不需要把代码放到固定的磁盘路径。
- DiscoRL 相关依赖通常需要安装：
  - `jax` + `jaxlib`
  - `disco_rl`（本仓库里的 `disco_rl-main` 也可作为本地源码使用）

## 运行方式（从项目根目录）

Stage-1 验证/训练：

```powershell
python models\validate.py
```

Stage-2 PPO：

```powershell
python ppo\train_stage2.py
```

Stage-2 DiscoRL：

```powershell
python ppo\train_stage2_disco.py
```

Stage-2 运行前提：

- `checkpoints/jscc_checkpoint.pth` 需要存在（Stage-2 会从这里读取 Stage-1 权重）。

