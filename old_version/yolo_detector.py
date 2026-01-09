"""
VALORANT YOLOv11 物体检测器
支持检测: enemy、head、teammate、item、flash
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Dict
import time

class YOLOv11Detector:
    """YOLOv11 ONNX模型检测器"""

    # 类别定义
    CLASS_NAMES = {
        0: "enemy",
        1: "head",
        2: "teammate",
        3: "item",
        4: "flash"
    }

    # 每个类别的颜色 (BGR格式)
    CLASS_COLORS = {
        0: (0, 0, 255),      # 红色 - enemy (敌人)
        1: (0, 0, 139),      # 深红色 - head (头部)
        2: (0, 255, 0),      # 绿色 - teammate (队友)
        3: (255, 255, 0),    # 青色 - item (道具)
        4: (0, 165, 255)     # 橙色 - flash (闪光)
    }

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化检测器

        Args:
            model_path: ONNX模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 加载模型
        print(f"[INFO] 加载模型: {Path(model_path).name}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 获取输入尺寸
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        print(f"[INFO] 模型输入尺寸: {self.input_width}x{self.input_height}")
        print(f"[INFO] 置信度阈值: {self.conf_threshold}")
        print(f"[INFO] NMS IOU阈值: {self.iou_threshold}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        预处理图像

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            处理后的图像张量, 缩放比例, padding偏移
        """
        # 保存原始尺寸
        original_height, original_width = image.shape[:2]

        # 计算缩放比例 (保持宽高比)
        scale = min(self.input_width / original_width, self.input_height / original_height)

        # 计算新的尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # 缩放图像
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 创建填充后的图像 (灰色背景)
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)

        # 计算padding偏移
        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2

        # 将缩放后的图像放到中心
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized

        # 转换为模型输入格式
        # BGR -> RGB
        image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # HWC -> CHW
        image_transposed = image_rgb.transpose(2, 0, 1)

        # 归一化到 [0, 1]
        image_normalized = image_transposed.astype(np.float32) / 255.0

        # 添加batch维度
        image_tensor = np.expand_dims(image_normalized, axis=0)

        return image_tensor, scale, (pad_x, pad_y)

    def postprocess(self, output: np.ndarray, scale: float, padding: Tuple[int, int],
                   original_shape: Tuple[int, int]) -> List[Dict]:
        """
        后处理模型输出

        Args:
            output: 模型输出 [1, 9, detections]
            scale: 缩放比例
            padding: padding偏移 (pad_x, pad_y)
            original_shape: 原始图像尺寸 (height, width)

        Returns:
            检测结果列表，每个检测包含: bbox, confidence, class_id, class_name
        """
        pad_x, pad_y = padding
        original_height, original_width = original_shape

        # 转置输出: [1, 9, detections] -> [detections, 9]
        predictions = output[0].transpose(1, 0)  # [detections, 9]

        detections = []

        for pred in predictions:
            # 前4个是边界框坐标 [x_center, y_center, width, height]
            x_center, y_center, width, height = pred[:4]

            # 后5个是类别置信度
            class_scores = pred[4:]

            # 获取最高置信度的类别
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            # 过滤低置信度检测
            if confidence < self.conf_threshold:
                continue

            # 将中心坐标转换为左上角坐标
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # 去除padding并缩放回原始尺寸
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # 限制在图像范围内
            x1 = max(0, min(x1, original_width))
            y1 = max(0, min(y1, original_height))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': self.CLASS_NAMES[class_id]
            })

        # 应用NMS (非极大值抑制)
        detections = self.nms(detections)

        return detections

    def nms(self, detections: List[Dict]) -> List[Dict]:
        """
        非极大值抑制

        Args:
            detections: 检测结果列表

        Returns:
            NMS后的检测结果
        """
        if len(detections) == 0:
            return []

        # 按类别分组进行NMS
        results = []
        for class_id in range(len(self.CLASS_NAMES)):
            # 过滤出当前类别的检测
            class_detections = [d for d in detections if d['class_id'] == class_id]

            if len(class_detections) == 0:
                continue

            # 提取边界框和置信度
            boxes = np.array([d['bbox'] for d in class_detections])
            scores = np.array([d['confidence'] for d in class_detections])

            # 使用OpenCV的NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.conf_threshold,
                self.iou_threshold
            )

            # 添加保留的检测
            if len(indices) > 0:
                for idx in indices.flatten():
                    results.append(class_detections[idx])

        return results

    def detect(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        检测图像中的目标

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            检测结果列表, 推理时间(秒)
        """
        # 保存原始尺寸
        original_shape = image.shape[:2]

        # 预处理
        input_tensor, scale, padding = self.preprocess(image)

        # 推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time

        # 后处理
        detections = self.postprocess(outputs[0], scale, padding, original_shape)

        return detections, inference_time

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            detections: 检测结果列表

        Returns:
            绘制后的图像
        """
        result_image = image.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            color = self.CLASS_COLORS[class_id]

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # 准备标签文本
            label = f"{class_name}: {confidence:.2f}"

            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 绘制标签背景
            cv2.rectangle(
                result_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # 绘制标签文本
            cv2.putText(
                result_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return result_image


def detect_image(model_path: str, image_path: str, output_path: str = None,
                conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    """
    检测单张图像

    Args:
        model_path: ONNX模型路径
        image_path: 输入图像路径
        output_path: 输出图像路径 (None则显示)
        conf_threshold: 置信度阈值
        iou_threshold: NMS IOU阈值
    """
    # 创建检测器
    detector = YOLOv11Detector(model_path, conf_threshold, iou_threshold)

    # 读取图像
    print(f"\n[INFO] 读取图像: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] 无法读取图像: {image_path}")
        return

    print(f"[INFO] 图像尺寸: {image.shape[1]}x{image.shape[0]}")

    # 检测
    print(f"[INFO] 开始检测...")
    detections, inference_time = detector.detect(image)

    # 输出结果
    print(f"\n[RESULT] 推理时间: {inference_time*1000:.2f}ms")
    print(f"[RESULT] 检测到 {len(detections)} 个目标:")

    # 统计每个类别的数量
    class_counts = {}
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} 个")

    # 打印详细信息
    print(f"\n[DETAIL] 检测详情:")
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det['bbox']
        print(f"  {i}. {det['class_name']} - 置信度: {det['confidence']:.3f}, "
              f"位置: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")

    # 绘制结果
    result_image = detector.draw_detections(image, detections)

    # 添加FPS信息
    fps = 1.0 / inference_time if inference_time > 0 else 0
    cv2.putText(
        result_image,
        f"FPS: {fps:.1f} | Detections: {len(detections)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # 保存或显示
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"\n[INFO] 结果已保存到: {output_path}")
    else:
        cv2.imshow("VALORANT Detection", result_image)
        print(f"\n[INFO] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """主函数"""
    print("="*80)
    print("VALORANT YOLOv11 物体检测器")
    print("="*80)

    # 配置
    model_path = "v11moudle/kenny_ultra_640_v11s.onnx"  # 使用640模型
    image_path = "test.png"
    output_path = "result.png"

    # 检测参数
    conf_threshold = 0.25  # 置信度阈值
    iou_threshold = 0.45   # NMS阈值

    # 检测图像
    detect_image(model_path, image_path, output_path, conf_threshold, iou_threshold)


if __name__ == "__main__":
    main()
