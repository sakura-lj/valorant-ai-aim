"""
VALORANT YOLOv11 物体检测器 - OpenVINO 版本
Intel GPU 优化，向量化后处理，深度性能优化
"""
import cv2
import numpy as np
from openvino.runtime import Core
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class YOLOv11DetectorOpenVINO:
    """YOLOv11 OpenVINO 检测器"""

    CLASS_NAMES = {
        0: "enemy",
        1: "head",
        2: "teammate",
        3: "item",
        4: "flash"
    }

    def __init__(self, model_path: str, device: str = "GPU",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 filter_class: Optional[str] = None):
        """
        初始化检测器

        Args:
            model_path: OpenVINO IR 模型路径 (.xml 文件)
            device: 推理设备 ("CPU", "GPU")
            conf_threshold: 置信度阈值
            iou_threshold: NMS 的 IOU 阈值
            filter_class: 只保留指定类别 (如 "head")，None 则保留所有类别
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.filter_class = filter_class

        # 初始化 OpenVINO
        self.core = Core()

        # 加载模型
        model = self.core.read_model(model=model_path)

        # GPU 优化配置
        if device == "GPU":
            config = {
                "PERFORMANCE_HINT": "LATENCY",
                "CACHE_DIR": "./cache"
            }
            self.compiled_model = self.core.compile_model(model, device, config)
        else:
            self.compiled_model = self.core.compile_model(model, device)

        # 获取输入输出信息
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # 获取输入尺寸
        input_shape = self.input_layer.shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        # 创建推理请求
        self.infer_request = self.compiled_model.create_infer_request()

        print(f"[INFO] 推理设备: {device}")
        print(f"[INFO] 模型输入尺寸: {self.input_width}x{self.input_height}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """预处理图像（真正优化版本）"""
        original_height, original_width = image.shape[:2]

        # 快速路径：尺寸完全匹配时，跳过 resize 和 padding
        if original_width == self.input_width and original_height == self.input_height:
            # 直接转换：BGR -> RGB -> CHW -> normalize
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_transposed = image_rgb.transpose(2, 0, 1)
            image_normalized = image_transposed.astype(np.float32) / 255.0
            image_tensor = np.expand_dims(image_normalized, axis=0)
            return image_tensor, 1.0, (0, 0)

        # 标准路径：尺寸不匹配时才需要 resize 和 padding
        scale = min(self.input_width / original_width, self.input_height / original_height)

        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 创建填充图像
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized

        # 转换为模型输入格式
        image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        image_transposed = image_rgb.transpose(2, 0, 1)
        image_normalized = image_transposed.astype(np.float32) / 255.0
        image_tensor = np.expand_dims(image_normalized, axis=0)

        return image_tensor, scale, (pad_x, pad_y)

    def postprocess(self, output: np.ndarray, scale: float, padding: Tuple[int, int],
                   original_shape: Tuple[int, int]) -> List[Dict]:
        """后处理模型输出（深度优化版本）"""
        pad_x, pad_y = padding
        original_height, original_width = original_shape

        # 转置输出
        predictions = output[0].transpose(1, 0)

        # 优化1：合并 argmax 和 max 操作
        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]  # 直接索引，避免 max

        # 过滤低置信度
        mask = confidences >= self.conf_threshold

        # 优化：类别过滤（只保留指定类别，大幅减少后续计算）
        if self.filter_class is not None:
            target_class_id = None
            for cid, cname in self.CLASS_NAMES.items():
                if cname == self.filter_class:
                    target_class_id = cid
                    break

            if target_class_id is not None:
                class_mask = class_ids == target_class_id
                mask = mask & class_mask

        if not mask.any():
            return []

        boxes = predictions[mask, :4]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # 批量转换坐标（向量化）
        x_centers = boxes[:, 0]
        y_centers = boxes[:, 1]
        widths = boxes[:, 2]
        heights = boxes[:, 3]

        # 优化2：使用就地操作减少临时数组
        half_w = widths * 0.5
        half_h = heights * 0.5

        x1 = x_centers - half_w
        y1 = y_centers - half_h
        x2 = x_centers + half_w
        y2 = y_centers + half_h

        # 批量去除 padding 并缩放
        if scale != 1.0 or pad_x != 0 or pad_y != 0:
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

        # 批量裁剪到图像范围
        x1 = np.clip(x1, 0, original_width)
        y1 = np.clip(y1, 0, original_height)
        x2 = np.clip(x2, 0, original_width)
        y2 = np.clip(y2, 0, original_height)

        # 优化3：向量化组装检测结果
        bboxes_array = np.stack([x1, y1, x2, y2], axis=1)

        detections = []
        for i in range(len(bboxes_array)):
            detections.append({
                'bbox': bboxes_array[i].tolist(),
                'confidence': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'class_name': self.CLASS_NAMES[int(class_ids[i])]
            })

        # 应用 NMS
        detections = self.nms(detections)

        return detections

    def nms(self, detections: List[Dict]) -> List[Dict]:
        """非极大值抑制（优化版本）"""
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
            bboxes = np.array([d['bbox'] for d in class_detections], dtype=np.float32)
            scores = np.array([d['confidence'] for d in class_detections], dtype=np.float32)

            # 转换为 [x, y, width, height] 格式
            boxes_xywh = np.zeros_like(bboxes)
            boxes_xywh[:, 0] = bboxes[:, 0]  # x
            boxes_xywh[:, 1] = bboxes[:, 1]  # y
            boxes_xywh[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
            boxes_xywh[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height

            # 使用 OpenCV 的 NMS
            indices = cv2.dnn.NMSBoxes(
                boxes_xywh.tolist(),
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
        original_shape = image.shape[:2]

        # 预处理
        input_tensor, scale, padding = self.preprocess(image)

        # 推理
        self.infer_request.infer({self.input_layer: input_tensor})
        output = self.infer_request.get_output_tensor(0).data

        # 后处理
        detections = self.postprocess(output, scale, padding, original_shape)

        return detections, None, None
