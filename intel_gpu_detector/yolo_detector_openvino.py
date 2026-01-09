import cv2
import numpy as np
from openvino.runtime import Core, Type, Layout, AsyncInferQueue
from openvino.preprocess import PrePostProcessor, ColorFormat
from typing import List, Tuple, Optional

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

        # --- 核心优化：集成预处理到模型中 ---
        ppp = PrePostProcessor(model)
        # 声明输入数据的格式 (用户提供的 BGR, HWC, uint8)
        ppp.input().tensor() \
            .set_element_type(Type.u8) \
            .set_layout(Layout("NHWC")) \
            .set_color_format(ColorFormat.BGR)
        
        # 声明模型预期的格式 (RGB, NCHW, float32)
        ppp.input().model().set_layout(Layout("NCHW"))
        
        # 在底层 C++ 中执行转换：BGR->RGB, u8->f32, /255.0
        ppp.input().preprocess() \
            .convert_element_type(Type.f32) \
            .convert_color(ColorFormat.RGB) \
            .scale(255.0)
        
        model = ppp.build()

        # CPU 性能配置 (针对 i3-10105F)
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": "4", # 4个物理核
            "AFFINITY": "CORE"
        }
        
        # 获取输入 Tensor 的 shape
        input_layer = self.compiled_model.input(0)
        input_shape = input_layer.shape

        # 针对 NCHW 布局的正确取值方式
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        print(f"模型输入尺寸确认: Width={self.input_width}, Height={self.input_height}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """极简预处理：只负责 Resize 和 Padding"""
        h, w = image.shape[:2]
        scale = min(self.input_width / w, self.input_height / h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # 预分配全黑图像并填充 (114 灰色)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        dx, dy = (self.input_width - nw) // 2, (self.input_height - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized
        
        # 注意：这里不再需要手动转置和归一化，由 OpenVINO PPP 完成
        return np.expand_dims(canvas, axis=0), scale, (dx, dy)

    def postprocess(self, output: np.ndarray, scale: float, padding: Tuple[int, int], 
                    img_shape: Tuple[int, int]) -> np.ndarray:
        """纯 NumPy 后处理，不产生中间 List"""
        # output shape: [1, 89, 21504] -> 转置并取前几个通道
        # YOLOv11 输出通常是 [1, 4+num_classes, num_boxes]
        preds = output[0].T 
        
        # 快速过滤置信度
        scores = preds[:, 4:]
        if self.target_class_id is not None:
            # 只看目标类别的分值，极大减少运算量
            conf = scores[:, self.target_class_id]
            class_ids = np.full(conf.shape, self.target_class_id)
        else:
            conf = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)

        mask = conf > self.conf_threshold
        if not np.any(mask): return np.empty((0, 6))

        # 提取过关的 boxes
        boxes = preds[mask, :4]
        conf = conf[mask]
        class_ids = class_ids[mask]

        # 坐标转换 [cx, cy, w, h] -> [x1, y1, x2, y2]
        half_w, half_h = boxes[:, 2] / 2, boxes[:, 3] / 2
        x1 = (boxes[:, 0] - half_w - padding[0]) / scale
        y1 = (boxes[:, 1] - half_h - padding[1]) / scale
        x2 = (boxes[:, 0] + half_w - padding[0]) / scale
        y2 = (boxes[:, 1] + half_h - padding[1]) / scale

        # 合并结果为矩阵: [x1, y1, x2, y2, conf, cls]
        res = np.column_stack([x1, y1, x2, y2, conf, class_ids])
        
        # NMS 过滤 (OpenCV NMSBoxes 依然很快)
        # NMS 需要 xywh 格式
        nms_boxes = np.column_stack([x1, y1, x2-x1, y2-y1])
        indices = cv2.dnn.NMSBoxes(nms_boxes.tolist(), conf.tolist(), self.conf_threshold, self.iou_threshold)
        
        return res[indices.flatten()] if len(indices) > 0 else np.empty((0, 6))

    def detect(self, image: np.ndarray):
        input_tensor, scale, pad = self.preprocess(image)
        self.infer_request.infer({0: input_tensor})
        output = self.infer_request.get_output_tensor(0).data
        return self.postprocess(output, scale, pad, image.shape[:2])