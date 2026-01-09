"""
VALORANT YOLOv11 物体检测器 - ONNX Runtime GPU 版本
支持所有 NVIDIA GPU，使用 ONNX Runtime CUDA 加速
"""
import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
from typing import List, Tuple, Dict
from collections import deque


class YOLOv11DetectorONNX:
    """YOLOv11 ONNX Runtime 检测器（GPU 加速）"""

    CLASS_NAMES = {
        0: "enemy",
        1: "head",
        2: "teammate",
        3: "item",
        4: "flash"
    }

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 enable_profiling: bool = False):
        """
        初始化检测器

        Args:
            model_path: ONNX 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS 的 IOU 阈值
            enable_profiling: 是否启用性能分析
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_profiling = enable_profiling

        # 性能监控
        if self.enable_profiling:
            self.perf_stats = {
                'preprocess': deque(maxlen=100),
                'inference': deque(maxlen=100),
                'postprocess': deque(maxlen=100),
                'nms': deque(maxlen=100),
                'total': deque(maxlen=100)
            }

        print(f"[INFO] 加载模型: {Path(model_path).name}")

        # 设置 ONNX Runtime 执行提供者（优先使用 CUDA）
        providers = []

        # 检查是否有 CUDA 支持
        available_providers = ort.get_available_providers()
        print(f"[INFO] 可用的执行提供者: {available_providers}")

        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            print(f"[INFO] 使用 CUDA 加速")
        else:
            print(f"[WARNING] CUDA 不可用，将使用 CPU")

        providers.append('CPUExecutionProvider')

        # 创建推理会话
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

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
        print(f"[INFO] 实际使用的提供者: {self.session.get_providers()}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        预处理图像

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            处理后的图像张量, 缩放比例, padding偏移
        """
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

        # 添加 batch 维度
        image_tensor = np.expand_dims(image_normalized, axis=0)

        return image_tensor, scale, (pad_x, pad_y)

    def postprocess(self, output: np.ndarray, scale: float, padding: Tuple[int, int],
                   original_shape: Tuple[int, int]) -> List[Dict]:
        """
        后处理模型输出（向量化优化版本）

        Args:
            output: 模型输出
            scale: 缩放比例
            padding: padding偏移
            original_shape: 原始图像尺寸

        Returns:
            检测结果列表
        """
        pad_x, pad_y = padding
        original_height, original_width = original_shape

        # 转置输出
        predictions = output[0].transpose(1, 0)

        # 向量化操作
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # 过滤低置信度
        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # 批量转换坐标
        x_centers = boxes[:, 0]
        y_centers = boxes[:, 1]
        widths = boxes[:, 2]
        heights = boxes[:, 3]

        x1 = x_centers - widths * 0.5
        y1 = y_centers - heights * 0.5
        x2 = x_centers + widths * 0.5
        y2 = y_centers + heights * 0.5

        # 批量去除 padding 并缩放
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # 批量裁剪到图像范围
        x1 = np.clip(x1, 0, original_width)
        y1 = np.clip(y1, 0, original_height)
        x2 = np.clip(x2, 0, original_width)
        y2 = np.clip(y2, 0, original_height)

        # 组装检测结果
        detections = []
        for i in range(len(boxes)):
            detections.append({
                'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                'confidence': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'class_name': self.CLASS_NAMES[int(class_ids[i])]
            })

        # 应用 NMS
        if self.enable_profiling:
            t_nms_start = time.perf_counter()
        detections = self.nms(detections)
        if self.enable_profiling:
            t_nms_end = time.perf_counter()
            self.perf_stats['nms'].append((t_nms_end - t_nms_start) * 1000)

        return detections

    def nms(self, detections: List[Dict]) -> List[Dict]:
        """
        非极大值抑制（优化版本）

        Args:
            detections: 检测结果列表

        Returns:
            NMS 后的检测结果
        """
        if len(detections) == 0:
            return []

        # 只处理有检测的类别
        unique_classes = set(d['class_id'] for d in detections)

        results = []
        for class_id in unique_classes:
            class_detections = [d for d in detections if d['class_id'] == class_id]

            # 单个检测，无需 NMS
            if len(class_detections) == 1:
                results.append(class_detections[0])
                continue

            # 提取边界框和置信度
            # 注意：bbox 存储的是 [x1, y1, x2, y2]，需要转换为 [x, y, width, height]
            boxes_xywh = []
            for d in class_detections:
                x1, y1, x2, y2 = d['bbox']
                boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

            boxes = np.array(boxes_xywh, dtype=np.float32)
            scores = np.array([d['confidence'] for d in class_detections], dtype=np.float32)

            # 使用 OpenCV 的 NMS
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

    def detect(self, image: np.ndarray) -> Tuple[List[Dict], None, None]:
        """
        检测图像中的目标

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            检测结果列表, None, None (保持接口兼容性)
        """
        if self.enable_profiling:
            t_start = time.perf_counter()

        original_shape = image.shape[:2]

        # 预处理
        if self.enable_profiling:
            t_pre_start = time.perf_counter()
        input_tensor, scale, padding = self.preprocess(image)
        if self.enable_profiling:
            t_pre_end = time.perf_counter()
            self.perf_stats['preprocess'].append((t_pre_end - t_pre_start) * 1000)

        # 推理
        if self.enable_profiling:
            t_inf_start = time.perf_counter()
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]
        if self.enable_profiling:
            t_inf_end = time.perf_counter()
            self.perf_stats['inference'].append((t_inf_end - t_inf_start) * 1000)

        # 后处理
        if self.enable_profiling:
            t_post_start = time.perf_counter()
        detections = self.postprocess(output, scale, padding, original_shape)
        if self.enable_profiling:
            t_post_end = time.perf_counter()
            self.perf_stats['postprocess'].append((t_post_end - t_post_start) * 1000)

        if self.enable_profiling:
            t_end = time.perf_counter()
            self.perf_stats['total'].append((t_end - t_start) * 1000)

        return detections, None, None

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取性能统计信息

        Returns:
            性能统计字典，包含每个步骤的平均、最小、最大耗时
        """
        if not self.enable_profiling:
            return {}

        stats = {}
        for key, values in self.perf_stats.items():
            if len(values) > 0:
                stats[key] = {
                    'avg': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values)
                }
        return stats

    def print_performance_stats(self):
        """打印性能统计信息"""
        if not self.enable_profiling:
            print("[INFO] 性能分析未启用")
            return

        stats = self.get_performance_stats()
        if not stats:
            print("[INFO] 暂无性能数据")
            return

        print("\n" + "="*80)
        print("性能分析报告 (基于最近 100 帧)")
        print("="*80)
        print(f"{'步骤':<15} {'平均(ms)':<12} {'最小(ms)':<12} {'最大(ms)':<12} {'标准差(ms)':<12}")
        print("-"*80)

        for key in ['preprocess', 'inference', 'postprocess', 'nms', 'total']:
            if key in stats:
                s = stats[key]
                print(f"{key:<15} {s['avg']:>10.2f}  {s['min']:>10.2f}  {s['max']:>10.2f}  {s['std']:>10.2f}")

        if 'total' in stats:
            avg_fps = 1000.0 / stats['total']['avg']
            print("-"*80)
            print(f"平均 FPS: {avg_fps:.1f}")
        print("="*80 + "\n")
