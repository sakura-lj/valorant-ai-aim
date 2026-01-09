"""
ONNX模型转换为TensorRT引擎
支持自动检测和转换多个YOLO模型
使用 TRTYOLO 库进行转换
"""
from pathlib import Path
import subprocess
import sys


def convert_onnx_to_tensorrt(onnx_path: str, output_dir: str = "models",
                             precision: str = "fp16"):
    """
    将ONNX模型转换为TensorRT引擎

    Args:
        onnx_path: ONNX模型路径
        output_dir: 输出目录
        precision: 精度模式 ("fp32", "fp16", "int8")

    Returns:
        bool: 转换是否成功
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)

    if not onnx_path.exists():
        print(f"[ERROR] ONNX模型不存在: {onnx_path}")
        return False

    # 创建输出目录
    output_dir.mkdir(exist_ok=True, parents=True)

    # 输出文件名 (去掉.onnx后缀)
    model_name = onnx_path.stem
    output_engine_path = output_dir / f"{model_name}.engine"

    print("="*80)
    print(f"转换模型: {onnx_path.name}")
    print("="*80)
    print(f"[INFO] 输入: {onnx_path}")
    print(f"[INFO] 输出: {output_engine_path}")
    print(f"[INFO] 精度: {precision.upper()}")

    try:
        print(f"\n[INFO] 开始转换ONNX模型为TensorRT引擎...")
        print(f"[INFO] 这可能需要几分钟时间，请耐心等待...")

        # 使用 trtyolo 命令行工具进行转换
        # 命令格式: trtyolo export -o model.onnx -e model.engine --precision fp16
        cmd = [
            sys.executable, "-m", "trtyolo", "export",
            "-o", str(onnx_path),
            "-e", str(output_engine_path),
            "--precision", precision
        ]

        print(f"\n[INFO] 执行命令: {' '.join(cmd)}\n")

        # 执行转换命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # 打印输出
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # 检查输出文件
        if output_engine_path.exists():
            # 获取文件大小
            engine_size = output_engine_path.stat().st_size / (1024 * 1024)  # MB

            print(f"\n[SUCCESS] 转换成功!")
            print(f"  - {output_engine_path} ({engine_size:.2f} MB)")
            return True
        else:
            print(f"\n[ERROR] 转换失败: 输出文件不存在")
            print(f"\n[提示] 请确保已安装 TRTYOLO:")
            print(f"  pip install trtyolo")
            print(f"\n[提示] 如果使用 trtyolo export 失败，可以尝试使用 trtexec:")
            print(f'  trtexec --onnx={onnx_path} --saveEngine={output_engine_path} --fp16')
            return False

    except Exception as e:
        print(f"\n[ERROR] 转换失败:")
        print(f"  {type(e).__name__}: {str(e)}")
        print(f"\n[提示] 请确保已安装 TRTYOLO:")
        print(f"  pip install trtyolo")
        return False


def batch_convert(onnx_dir: str = "./v11moudle", output_dir: str = "models",
                 precision: str = "fp16"):
    """
    批量转换目录中的所有ONNX模型

    Args:
        onnx_dir: ONNX模型目录
        output_dir: 输出目录
        precision: 精度模式 ("fp32", "fp16", "int8")
    """
    onnx_dir = Path(onnx_dir)

    if not onnx_dir.exists():
        print(f"[ERROR] 目录不存在: {onnx_dir}")
        return

    # 查找所有ONNX文件
    onnx_files = list(onnx_dir.glob("*.onnx"))

    if len(onnx_files) == 0:
        print(f"[WARNING] 未找到ONNX模型文件在: {onnx_dir}")
        return

    print(f"\n找到 {len(onnx_files)} 个ONNX模型:")
    for i, onnx_file in enumerate(onnx_files, 1):
        print(f"  {i}. {onnx_file.name}")

    print(f"\n开始批量转换...\n")

    success_count = 0
    for onnx_file in onnx_files:
        if convert_onnx_to_tensorrt(str(onnx_file), output_dir, precision):
            success_count += 1
        print()  # 空行分隔

    print("="*80)
    print(f"转换完成: {success_count}/{len(onnx_files)} 个模型成功")
    print("="*80)


def convert_using_trtexec(onnx_path: str, output_dir: str = "models",
                          precision: str = "fp16"):
    """
    使用 trtexec 工具转换ONNX模型为TensorRT引擎
    这是 NVIDIA 官方工具，通常随 TensorRT 一起安装

    Args:
        onnx_path: ONNX模型路径
        output_dir: 输出目录
        precision: 精度模式 ("fp32", "fp16", "int8")

    Returns:
        bool: 转换是否成功
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)

    if not onnx_path.exists():
        print(f"[ERROR] ONNX模型不存在: {onnx_path}")
        return False

    # 创建输出目录
    output_dir.mkdir(exist_ok=True, parents=True)

    # 输出文件名
    model_name = onnx_path.stem
    output_engine_path = output_dir / f"{model_name}.engine"

    print("="*80)
    print(f"使用 trtexec 转换模型: {onnx_path.name}")
    print("="*80)
    print(f"[INFO] 输入: {onnx_path}")
    print(f"[INFO] 输出: {output_engine_path}")
    print(f"[INFO] 精度: {precision.upper()}")

    try:
        # 构建 trtexec 命令
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={output_engine_path}",
        ]

        # 添加精度参数
        if precision == "fp16":
            cmd.append("--fp16")
        elif precision == "int8":
            cmd.append("--int8")

        print(f"\n[INFO] 执行命令: {' '.join(cmd)}\n")

        # 执行转换命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # 打印输出
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # 检查输出文件
        if output_engine_path.exists():
            engine_size = output_engine_path.stat().st_size / (1024 * 1024)
            print(f"\n[SUCCESS] 转换成功!")
            print(f"  - {output_engine_path} ({engine_size:.2f} MB)")
            return True
        else:
            print(f"\n[ERROR] 转换失败: 输出文件不存在")
            return False

    except FileNotFoundError:
        print(f"\n[ERROR] 未找到 trtexec 工具")
        print(f"[提示] trtexec 通常随 TensorRT 一起安装")
        print(f"[提示] 请确保 TensorRT 已正确安装并添加到 PATH")
        return False
    except Exception as e:
        print(f"\n[ERROR] 转换失败:")
        print(f"  {type(e).__name__}: {str(e)}")
        return False


def main():
    """主函数"""
    print("="*80)
    print("ONNX → TensorRT 引擎转换工具")
    print("="*80)

    # 方式1: 使用 TRTYOLO 转换单个模型
    # convert_onnx_to_tensorrt("../v11moudle/val_kenny_ultra_256_v11s.onnx", "models", precision="fp16")

    # 方式2: 批量转换目录中的所有模型 (推荐)
    batch_convert("../v11moudle", "models", precision="fp16")

    # 方式3: 使用 trtexec 转换（备用方案）
    # convert_using_trtexec("../v11moudle/val_kenny_ultra_256_v11s.onnx", "models", precision="fp16")

    print("\n[INFO] 转换提示:")
    print("  - FP16 精度: 推荐用于大多数情况，速度快且精度损失小")
    print("  - FP32 精度: 最高精度，但速度较慢")
    print("  - INT8 精度: 最快速度，但需要校准数据集")


if __name__ == "__main__":
    main()
