"""
TensorRT 引擎诊断脚本
用于检查 TRTYOLO 和引擎文件的兼容性
"""
import os
import sys

# 添加 DLL 路径
if sys.platform == 'win32':
    base_trt = r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.14.1.48"
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"

    extra_paths = [
        os.path.join(base_trt, "bin"),
        os.path.join(base_trt, "lib"),
        cuda_bin
    ]

    for p in extra_paths:
        if os.path.exists(p):
            os.add_dll_directory(p)

print("="*80)
print("TensorRT 引擎诊断工具")
print("="*80)

# 1. 检查 TRTYOLO 版本
print("\n[1] 检查 TRTYOLO 安装...")
try:
    import trtyolo
    print(f"✓ TRTYOLO 已安装")
    print(f"  版本: {trtyolo.__version__ if hasattr(trtyolo, '__version__') else '未知'}")
    print(f"  路径: {trtyolo.__file__}")
except ImportError as e:
    print(f"✗ TRTYOLO 未安装: {e}")
    sys.exit(1)

# 2. 检查 TRTYOLO 的初始化签名
print("\n[2] 检查 TRTYOLO 类...")
from trtyolo import TRTYOLO
import inspect

sig = inspect.signature(TRTYOLO.__init__)
print(f"✓ TRTYOLO.__init__ 参数签名:")
print(f"  {sig}")

# 3. 检查引擎文件
print("\n[3] 检查引擎文件...")
engine_path = "../models/yolo11s_640.engine"

if os.path.exists(engine_path):
    file_size = os.path.getsize(engine_path) / (1024 * 1024)  # MB
    print(f"✓ 引擎文件存在: {engine_path}")
    print(f"  大小: {file_size:.2f} MB")
else:
    print(f"✗ 引擎文件不存在: {engine_path}")
    sys.exit(1)

# 4. 尝试不同的初始化方式
print("\n[4] 测试 TRTYOLO 初始化...")

test_cases = [
    {
        "name": "方式1: 位置参数 + task",
        "args": [engine_path],
        "kwargs": {"task": "detect", "profile": False, "swap_rb": True}
    },
    {
        "name": "方式2: 仅位置参数",
        "args": [engine_path],
        "kwargs": {"profile": False, "swap_rb": True}
    },
    {
        "name": "方式3: 关键字 model",
        "args": [],
        "kwargs": {"model": engine_path, "task": "detect", "profile": False, "swap_rb": True}
    },
    {
        "name": "方式4: 最简单",
        "args": [engine_path],
        "kwargs": {}
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"\n  测试 {i}: {test['name']}")
    try:
        model = TRTYOLO(*test['args'], **test['kwargs'])
        print(f"  ✓ 成功！")
        print(f"    使用此方式: TRTYOLO({test['args']}, {test['kwargs']})")

        # 尝试一个简单的预测
        import numpy as np
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = model.predict(test_img)
        print(f"  ✓ 预测测试成功")
        print(f"    结果类型: {type(result)}")
        if hasattr(result, '__dict__'):
            print(f"    结果属性: {list(result.__dict__.keys())}")

        del model
        break
    except Exception as e:
        print(f"  ✗ 失败: {type(e).__name__}: {str(e)}")

print("\n" + "="*80)
print("诊断完成")
print("="*80)
