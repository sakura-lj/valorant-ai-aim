"""
使用 TRTYOLO 将 ONNX 模型转换为 TensorRT 引擎
简化版本，专门用于 YOLOv11 检测模型
"""
import os
import sys
from pathlib import Path

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
print("ONNX → TensorRT 引擎转换工具 (使用 TRTYOLO)")
print("="*80)

# 配置
onnx_path = "../v11moudle/kenny_ultra_640_v11s.onnx"  # 使用 640 模型
output_path = "../models/yolo11s_640.engine"
fp16 = True  # 使用 FP16 精度

print(f"\n[配置]")
print(f"  输入 ONNX: {onnx_path}")
print(f"  输出引擎: {output_path}")
print(f"  精度模式: {'FP16' if fp16 else 'FP32'}")

# 检查输入文件
if not os.path.exists(onnx_path):
    print(f"\n[ERROR] ONNX 文件不存在: {onnx_path}")
    sys.exit(1)

onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"  ONNX 大小: {onnx_size:.2f} MB")

# 创建输出目录
output_dir = Path(output_path).parent
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[开始转换]")
print(f"  这可能需要几分钟时间，请耐心等待...")
print(f"  (TensorRT 正在针对您的 GPU 优化模型)\n")

try:
    # 方法1: 使用 trtyolo 命令行工具
    import subprocess

    cmd = [
        sys.executable, "-m", "trtyolo", "export",
        "-o", onnx_path,
        "-e", output_path,
    ]

    if fp16:
        cmd.append("--fp16")

    print(f"[INFO] 执行命令:")
    print(f"  {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=False,  # 直接显示输出
        text=True,
        check=False
    )

    # 检查结果
    if os.path.exists(output_path):
        engine_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n" + "="*80)
        print(f"[SUCCESS] 转换成功！")
        print(f"="*80)
        print(f"  引擎文件: {output_path}")
        print(f"  引擎大小: {engine_size:.2f} MB")
        print(f"\n现在可以运行: python simple_detector.py")
    else:
        print(f"\n" + "="*80)
        print(f"[ERROR] 转换失败：输出文件不存在")
        print(f"="*80)
        print(f"\n请检查上面的错误信息")
        sys.exit(1)

except Exception as e:
    print(f"\n[ERROR] 转换过程出错:")
    print(f"  {type(e).__name__}: {str(e)}")
    print(f"\n[提示] 请确保:")
    print(f"  1. CUDA 和 TensorRT 已正确安装")
    print(f"  2. GPU 驱动是最新版本")
    print(f"  3. 有足够的磁盘空间（至少 500MB）")
    sys.exit(1)
