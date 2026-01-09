"""
ONNX模型转换为OpenVINO IR格式
支持自动检测和转换多个YOLO模型
使用 OpenVINO Python API
"""
import openvino as ov
from pathlib import Path
import numpy as np


def convert_onnx_to_openvino(onnx_path: str, output_dir: str = "models",
                            compress_to_fp16: bool = True):
    """
    将ONNX模型转换为OpenVINO IR格式 (使用Python API)

    Args:
        onnx_path: ONNX模型路径
        output_dir: 输出目录
        compress_to_fp16: 是否压缩为FP16精度 (Intel GPU推荐)

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
    output_xml_path = output_dir / f"{model_name}.xml"
    output_bin_path = output_dir / f"{model_name}.bin"

    print("="*80)
    print(f"转换模型: {onnx_path.name}")
    print("="*80)
    print(f"[INFO] 输入: {onnx_path}")
    print(f"[INFO] 输出: {output_xml_path}")
    print(f"[INFO] 精度: {'FP16' if compress_to_fp16 else 'FP32'}")

    try:
        print(f"\n[INFO] 正在读取ONNX模型...")

        # 使用 OpenVINO Python API 转换模型
        model = ov.convert_model(str(onnx_path))

        print(f"[INFO] 模型转换成功")
        print(f"[INFO] 输入形状: {model.inputs[0].get_partial_shape()}")
        print(f"[INFO] 输出形状: {model.outputs[0].get_partial_shape()}")

        # 序列化保存模型
        print(f"\n[INFO] 正在保存OpenVINO IR模型...")
        ov.save_model(model, str(output_xml_path), compress_to_fp16=compress_to_fp16)

        # 检查输出文件
        if output_xml_path.exists() and output_bin_path.exists():
            # 获取文件大小
            xml_size = output_xml_path.stat().st_size / (1024 * 1024)  # MB
            bin_size = output_bin_path.stat().st_size / (1024 * 1024)  # MB

            print(f"\n[SUCCESS] 转换成功!")
            print(f"  - {output_xml_path} ({xml_size:.2f} MB)")
            print(f"  - {output_bin_path} ({bin_size:.2f} MB)")
            print(f"  - 总大小: {xml_size + bin_size:.2f} MB")
            return True
        else:
            print(f"\n[ERROR] 转换失败: 输出文件不存在")
            return False

    except Exception as e:
        print(f"\n[ERROR] 转换失败:")
        print(f"  {type(e).__name__}: {str(e)}")
        print(f"\n[提示] 请确保已安装 OpenVINO:")
        print(f"  pip install openvino openvino-dev")
        return False


def batch_convert(onnx_dir: str = "./v11moudle", output_dir: str = "models",
                 compress_to_fp16: bool = True):
    """
    批量转换目录中的所有ONNX模型

    Args:
        onnx_dir: ONNX模型目录
        output_dir: 输出目录
        compress_to_fp16: 是否压缩为FP16精度
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
        if convert_onnx_to_openvino(str(onnx_file), output_dir, compress_to_fp16):
            success_count += 1
        print()  # 空行分隔

    print("="*80)
    print(f"转换完成: {success_count}/{len(onnx_files)} 个模型成功")
    print("="*80)


def main():
    """主函数"""
    print("="*80)
    print("ONNX → OpenVINO IR 模型转换工具 (Python API)")
    print("="*80)

    # 方式1: 转换单个模型
    # convert_onnx_to_openvino("../v11moudle/kenny_ultra_640_v11s.onnx", "models", compress_to_fp16=True)

    # 方式2: 批量转换目录中的所有模型 (推荐)
    batch_convert("../v11moudle", "models", compress_to_fp16=True)


if __name__ == "__main__":
    main()
