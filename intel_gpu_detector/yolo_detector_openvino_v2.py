"""
VALORANT YOLOv11 物体检测器 - OpenVINO 深度优化版
针对 i3-10105F CPU 极致优化
核心优化：OpenVINO PPP + 纯 NumPy 流水线 + CPU 线程绑定
"""
import cv2
import numpy as np
from openvino.runtime import Core, Type, Layout
from openvino.preprocess import PrePostProcessor, ColorFormat
from typing import Optional

class YOLOv11DetectorOpenVINO:
    CLASS_NAMES = {0: "enemy", 1: "head", 2: "teammate", 3: "item", 4: "flash"}

    def __init__(self, model_path: str, device: str = "CPU",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 filter_class: Optional[str] = None):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.filter_class = filter_class
        self.target_class_id = next((k for k, v in self.CLASS_NAMES.items() if v == filter_class), None)

        self.core = Core()
        model = self.core.read_model(model=model_path)

        # ============ 核心优化 1: OpenVINO Preprocessing Pipeline ============
        # 将 BGR->RGB, uint8->float32, /255.0 下推到 C++ 层执行
        ppp = PrePostProcessor(model)

        # 声明输入数据格式（用户提供的 BGR, NHWC, uint8）
        ppp.input().tensor() \
            .set_element_type(Type.u8) \
            .set_layout(Layout("NHWC")) \
            .set_color_format(ColorFormat.BGR)

        # 声明模型期望格式（RGB, NCHW, float32）
        ppp.input().model().set_layout(Layout("NCHW"))

        # 自动插入转换操作（在 C++ 底层执行，快 3-5 倍）
        ppp.input().preprocess() \
            .convert_element_type(Type.f32) \
            .convert_color(ColorFormat.RGB) \
            .scale(255.0)

        model = ppp.build()

        # ============ 核心优化 2: CPU 线程绑定 ============
        # 针对 i3-10105F (4 核 8 线程) 优化
        if device == "CPU":
            config = {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_NUM_THREADS": "4",  # 使用 4 个物理核
                "AFFINITY": "CORE"  # 绑定物理核，避免超线程竞争
            }
            self.compiled_model = self.core.compile_model(model, device, config)
        else:
            self.compiled_model = self.core.compile_model(model, device)

        self.infer_request = self.compiled_model.create_infer_request()

        # 获取输入尺寸（注意：虽然设置了 NHWC，但 shape 还是 NCHW 格式）
        input_shape = self.compiled_model.input(0).shape  # [1, 3, H, W]
        self.input_height = input_shape[2]  # ✅ 修正：使用 [2] 和 [3]
        self.input_width = input_shape[3]

        print(f"[INFO] 推理设备: {device}")
        print(f"[INFO] 模型输入尺寸: {self.input_width}x{self.input_height}")
        if self.target_class_id is not None:
            print(f"[INFO] 类别过滤: 只检测 '{filter_class}' (ID={self.target_class_id})")

    def preprocess(self, image: np.ndarray):
        """
        极简预处理：只负责 Resize 和 Padding
        颜色转换和归一化由 OpenVINO PPP 自动完成
        """
        h, w = image.shape[:2]

        # 快速路径：尺寸完全匹配时，跳过 resize
        if w == self.input_width and h == self.input_height:
            return np.expand_dims(image, axis=0), 1.0, (0, 0)

        # 计算缩放比例（保持宽高比）
        scale = min(self.input_width / w, self.input_height / h)
        nw, nh = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # 创建填充画布（灰色 114）
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        dx, dy = (self.input_width - nw) // 2, (self.input_height - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized

        # 注意：不需要手动 BGR->RGB 和归一化，PPP 会自动处理
        return np.expand_dims(canvas, axis=0), scale, (dx, dy)

    def postprocess(self, output: np.ndarray, scale: float, padding: tuple,
                    img_shape: tuple) -> np.ndarray:
        """
        ============ 核心优化 3: 纯 NumPy 后处理 ============
        不创建 Python 对象，全程向量化操作
        返回 NumPy 数组: [[x1, y1, x2, y2, conf, class_id], ...]
        """
        # 提前检查空结果
        if output.size == 0:
            return np.empty((0, 6))

        # output shape: [1, 4+num_classes, num_boxes]
        # 转置为: [num_boxes, 4+num_classes]
        preds = output[0].T

        if len(preds) == 0:
            return np.empty((0, 6))

        # 提取类别分数
        scores = preds[:, 4:]

        # ============ 核心优化 4: 类别过滤优化 ============
        # 如果只检测一个类别，只计算该类别的置信度（减少 80% 计算量）
        if self.target_class_id is not None:
            conf = scores[:, self.target_class_id]
            class_ids = np.full(len(conf), self.target_class_id, dtype=np.int32)
        else:
            conf = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)

        # 置信度过滤
        mask = conf >= self.conf_threshold
        if not np.any(mask):
            return np.empty((0, 6))

        boxes = preds[mask, :4]
        conf = conf[mask]
        class_ids = class_ids[mask]

        # ============ 核心优化 5: 向量化坐标转换 ============
        # [cx, cy, w, h] -> [x1, y1, x2, y2]
        half_w, half_h = boxes[:, 2] * 0.5, boxes[:, 3] * 0.5
        x1 = (boxes[:, 0] - half_w - padding[0]) / scale
        y1 = (boxes[:, 1] - half_h - padding[1]) / scale
        x2 = (boxes[:, 0] + half_w - padding[0]) / scale
        y2 = (boxes[:, 1] + half_h - padding[1]) / scale

        # 裁剪到图像边界
        x1 = np.clip(x1, 0, img_shape[1])
        y1 = np.clip(y1, 0, img_shape[0])
        x2 = np.clip(x2, 0, img_shape[1])
        y2 = np.clip(y2, 0, img_shape[0])

        # 合并为 [x1, y1, x2, y2, conf, class_id]
        detections = np.column_stack([x1, y1, x2, y2, conf, class_ids])

        # NMS（使用 OpenCV 的高效实现）
        if len(detections) > 1:
            nms_boxes = np.column_stack([x1, y1, x2-x1, y2-y1])  # xywh 格式
            indices = cv2.dnn.NMSBoxes(
                nms_boxes.tolist(),
                conf.tolist(),
                self.conf_threshold,
                self.iou_threshold
            )
            if len(indices) > 0:
                detections = detections[indices.flatten()]

        return detections

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        检测图像中的目标

        Args:
            image: 输入图像 (BGR, uint8)

        Returns:
            NumPy 数组: [[x1, y1, x2, y2, conf, class_id], ...]
        """
        # 预处理
        input_tensor, scale, padding = self.preprocess(image)

        # 推理
        self.infer_request.infer({0: input_tensor})
        output = self.infer_request.get_output_tensor(0).data

        # 后处理
        return self.postprocess(output, scale, padding, image.shape[:2])
