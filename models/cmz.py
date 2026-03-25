import sys
import numpy as np

try:
    import jax
    print(f"✅ JAX 成功: {jax.__version__}")
    import torch
    print(f"✅ Torch 成功: {torch.__version__}")
    import matplotlib
    print(f"✅ Matplotlib 成功: {matplotlib.__version__}")
    print(f"📌 当前 Python 版本: {sys.version}")
    print(f"📌 当前 NumPy 版本: {np.__version__}")
except Exception as e:
    print(f"❌ 依然存在冲突: {e}")