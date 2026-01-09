"""
ONNX模型分析脚本
用于检查YOLOv11 ONNX模型的输入输出结构
"""
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

def analyze_onnx_model(model_path):
    """分析ONNX模型的详细信息"""
    print(f"\n{'='*80}")
    print(f"分析模型: {Path(model_path).name}")
    print(f"{'='*80}")

    # 使用onnx库加载模型
    try:
        model = onnx.load(model_path)
        print("\n[OK] 使用onnx库成功加载模型")

        # 获取模型的输入信息
        print("\n【输入信息】")
        for input_tensor in model.graph.input:
            print(f"  名称: {input_tensor.name}")
            shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})"
                    for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  形状: {shape}")
            dtype = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
            print(f"  数据类型: {dtype}")

        # 获取模型的输出信息
        print("\n【输出信息】")
        for i, output_tensor in enumerate(model.graph.output):
            print(f"\n  输出 #{i}:")
            print(f"    名称: {output_tensor.name}")
            shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})"
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"    形状: {shape}")
            dtype = onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)
            print(f"    数据类型: {dtype}")

    except Exception as e:
        print(f"\n[ERROR] onnx库加载失败: {e}")

    # 使用onnxruntime加载模型
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("\n[OK] 使用onnxruntime成功创建推理会话")

        # 获取输入信息
        print("\n【ONNXRuntime - 输入信息】")
        for input_meta in session.get_inputs():
            print(f"  名称: {input_meta.name}")
            print(f"  形状: {input_meta.shape}")
            print(f"  数据类型: {input_meta.type}")

        # 获取输出信息
        print("\n【ONNXRuntime - 输出信息】")
        for i, output_meta in enumerate(session.get_outputs()):
            print(f"\n  输出 #{i}:")
            print(f"    名称: {output_meta.name}")
            print(f"    形状: {output_meta.shape}")
            print(f"    数据类型: {output_meta.type}")

        # 尝试用随机数据进行推理测试
        print("\n【推理测试】")
        input_shape = session.get_inputs()[0].shape
        # 处理动态维度
        test_shape = []
        for dim in input_shape:
            if isinstance(dim, str) or dim <= 0:
                test_shape.append(1)  # 动态维度设为1
            else:
                test_shape.append(dim)

        print(f"  测试输入形状: {test_shape}")
        dummy_input = np.random.randn(*test_shape).astype(np.float32)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy_input})

        print(f"  [OK] 推理成功!")
        print(f"  输出数量: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  输出 #{i} 形状: {output.shape}, 数据类型: {output.dtype}")

            # 分析输出维度含义（针对YOLOv11）
            if len(output.shape) == 3:
                batch, elements, detections = output.shape
                print(f"    -> Batch: {batch}, Elements: {elements}, Detections: {detections}")
                print(f"    -> 预测: 每张图像有 {detections} 个可能的检测")
                print(f"    -> 每个检测有 {elements} 个元素")
                print(f"    -> 通常格式: [x, y, w, h, conf_class0, conf_class1, ..., conf_classN]")
                print(f"    -> 对于5个类别: 4 (bbox) + 5 (classes) = 9 个元素")
            elif len(output.shape) == 2:
                detections, elements = output.shape
                print(f"    -> Detections: {detections}, Elements: {elements}")

    except Exception as e:
        print(f"\n[ERROR] onnxruntime推理失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数：分析所有ONNX模型"""
    model_dir = Path("v11moudle")

    if not model_dir.exists():
        print(f"错误: 目录 {model_dir} 不存在")
        return

    # 获取所有ONNX模型
    onnx_files = list(model_dir.glob("*.onnx"))

    if not onnx_files:
        print(f"错误: 在 {model_dir} 中没有找到ONNX文件")
        return

    print(f"\n找到 {len(onnx_files)} 个ONNX模型文件")
    print(f"检测类别: 0-头, 1-敌人, 2-队友, 3-道具, 4-闪光")

    # 分析每个模型
    for model_path in sorted(onnx_files):
        analyze_onnx_model(str(model_path))

    print(f"\n{'='*80}")
    print("分析完成!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
