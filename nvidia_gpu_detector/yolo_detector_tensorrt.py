"""
VALORANT YOLOv11 物体检测器 - TensorRT 版本
NVIDIA GPU 优化，使用 TRTYOLO 推理引擎
"""

import cv2
import numpy as np
import os
import sys
# ==================== 1. 核心修复：环境初始化 ====================
if sys.platform == 'win32':
    # 注意：TensorRT 的 .dll 文件通常在 bin 目录下，.lib 在 lib 目录下
    # 我们需要将所有包含动态库的路径都加入
    base_trt = r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.14.1.48"
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
    
    extra_paths = [
        os.path.join(base_trt, "bin"), # 存放 nvinfer_10.dll
        os.path.join(base_trt, "lib"), # 存放部分依赖库
        cuda_bin                       # 存放 cudart64_xx.dll
    ]
    
    for p in extra_paths:
        if os.path.exists(p):
            os.add_dll_directory(p)
            # print(f"[DEBUG] 已加载依赖路径: {p}")
            
from trtyolo import TRTYOLO
from typing import List, Tuple, Dict


class YOLOv11DetectorTensorRT:
    """YOLOv11 TensorRT 检测器"""

    CLASS_NAMES = {
        0: "enemy",
        1: "head",
        2: "teammate",
        3: "item",
        4: "flash"
    }

    def __init__(self, engine_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 检查引擎文件是否存在
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"[ERROR] TensorRT 引擎文件不存在: {engine_path}")

        print(f"[INFO] 正在加载引擎: {engine_path}")

        try:
            # 初始化模型
            # 参考示例: TRTYOLO(args.engine, task="detect", profile=True, swap_rb=True)
            # 第一个参数是引擎路径（位置参数），后面是关键字参数
            self.model = TRTYOLO(
                str(engine_path),   # 第一个位置参数：引擎文件路径
                task="detect",      # 任务类型：检测
                profile=False,      # 不打印性能分析
                swap_rb=True        # 自动处理 BGR 到 RGB
            )
            print(f"[INFO] TensorRT 引擎加载成功")
            print(f"[INFO] 置信度阈值: {self.conf_threshold}")
            print(f"[INFO] NMS IOU阈值: {self.iou_threshold}")
        except Exception as e:
            print(f"[ERROR] 加载 TensorRT 引擎失败:")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误信息: {str(e)}")
            print(f"\n[提示] 请检查:")
            print(f"  1. 引擎文件是否在当前 GPU 上生成")
            print(f"  2. TensorRT 版本是否匹配")
            print(f"  3. CUDA 和 cuDNN 是否正确安装")
            raise

    def detect(self, image: np.ndarray) -> Tuple[List[Dict], None, None]:
        """
        检测图像中的目标

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            检测结果列表, None, None (保持接口兼容性)
        """
        # TRTYOLO 推理
        # result 是一个 Detection 对象，包含 bbox, class_id, confidence 等属性
        result = self.model.predict(image)

        # 转换为标准格式
        detections = []

        # 检查是否有检测结果
        if result and hasattr(result, 'xyxy') and len(result.xyxy) > 0:
            # result.xyxy: [N, 4] 边界框坐标 (x1, y1, x2, y2)
            # result.class_id: [N] 类别 ID
            # result.confidence: [N] 置信度

            for i in range(len(result.xyxy)):
                bbox = result.xyxy[i]
                class_id = int(result.class_id[i])
                confidence = float(result.confidence[i])

                # 过滤低置信度
                if confidence < self.conf_threshold:
                    continue

                detections.append({
                    'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.CLASS_NAMES.get(class_id, f"class_{class_id}")
                })

        return detections, None, None
