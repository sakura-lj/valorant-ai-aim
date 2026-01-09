import cv2
import numpy as np
import dxcam
import time
from collections import deque
from typing import Tuple, Optional
from yolo_detector_openvino import YOLOv11DetectorOpenVINO

class SimpleRealtimeDetector:
    """极简、深度优化的实时检测器"""

    def __init__(self,
                 model_path: str,
                 center_size: int = 256,
                 device: str = "CPU",
                 conf_threshold: float = 0.60,
                 iou_threshold: float = 0.40,
                 filter_class: str = "head"):
        """
        初始化检测器
        """
        print("="*80)
        print(f"VALORANT 实时检测系统 - 深度优化版 (捕获中心 {center_size}x{center_size})")
        print("="*80)

        self.center_size = center_size
        self.screen_cx = center_size // 2
        self.screen_cy = center_size // 2

        # 1. 加载优化后的推理引擎
        print("\n[1/2] 加载 OpenVINO 推理引擎 (向量化版本)...")
        self.detector = YOLOv11DetectorOpenVINO(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            filter_class=filter_class
        )

        # 2. 计算并初始化捕获区域
        print("\n[2/2] 初始化 DXCam 捕获...")
        self.capture_region = self._get_center_region()
        self.camera = dxcam.create(output_color="BGR")
        # 预先启动摄像头，设置极高的 target_fps 避免等待
        self.camera.start(region=self.capture_region, target_fps=200)

        # FPS 计数
        self.frame_times = deque(maxlen=30)
        self.fps = 0

        print("\n[SUCCESS] 系统初始化完成! 准备开始检测...")
        print("="*80)

    def _get_center_region(self) -> Tuple[int, int, int, int]:
        """计算屏幕中心区域坐标"""
        # 临时创建以获取屏幕尺寸
        tmp = dxcam.create()
        screen_width, screen_height = tmp.width, tmp.height
        del tmp

        left = (screen_width - self.center_size) // 2
        top = (screen_height - self.center_size) // 2
        right = left + self.center_size
        bottom = top + self.center_size

        return (left, top, right, bottom)

    def find_best_target(self, detections: np.ndarray):
        """
        使用纯 NumPy 向量化寻找距离中心最近的目标
        detections 格式: [[x1, y1, x2, y2, conf, cls], ...]
        """
        if len(detections) == 0:
            return None

        # 1. 提取所有目标的中心坐标
        # x_centers = (x1 + x2) / 2, y_centers = (y1 + y2) / 2
        centers = (detections[:, 0:2] + detections[:, 2:4]) * 0.5
        
        # 2. 计算到屏幕中心 (screen_cx, screen_cy) 的欧氏距离
        # 使用广播机制计算: sqrt((dx)^2 + (dy)^2)
        diffs = centers - np.array([self.screen_cx, self.screen_cy])
        distances = np.linalg.norm(diffs, axis=1)

        # 3. 找到距离最小的索引
        idx = np.argmin(distances)
        
        return {
            'bbox': detections[idx, 0:4],
            'conf': detections[idx, 4],
            'dist': distances[idx],
            'ox': diffs[idx, 0],
            'oy': diffs[idx, 1]
        }

    def run(self):
        """主循环：抓图 -> 检测 -> 渲染"""
        print("\n检测运行中... 按 'Q' 键在视频窗口退出")
        
        try:
            while True:
                t1 = time.perf_counter()

                # 1. 快速抓图
                frame = self.camera.get_latest_frame()
                if frame is None: continue

                # 2. 推理 (返回 NumPy 数组，不产生 Python 对象)
                detections = self.detector.detect(frame)

                # 3. 寻找目标
                best_target = self.find_best_target(detections)

                # 4. 可视化渲染 (优化：仅在有窗口需求时执行)
                self._render(frame, detections, best_target)

                # 5. 性能统计
                self.frame_times.append(time.perf_counter() - t1)
                if len(self.frame_times) >= 30:
                    self.fps = 1.0 / (sum(self.frame_times) / 30)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

    def _render(self, frame, detections, target):
        """轻量级可视化，减少绘制开销"""
        # 绘制中心准星
        cv2.drawMarker(frame, (self.screen_cx, self.screen_cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)

        # 绘制检测到的所有 head
        for det in detections:
            x1, y1, x2, y2 = det[0:4].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # 绘制锁定连线
        if target:
            tx, ty = int((target['bbox'][0] + target['bbox'][2]) / 2), \
                     int((target['bbox'][1] + target['bbox'][3]) / 2)
            cv2.line(frame, (self.screen_cx, self.screen_cy), (tx, ty), (255, 255, 0), 1)
            
            # 显示距离信息
            cv2.putText(frame, f"DIST: {target['dist']:.1f}", (10, self.center_size - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 标题栏实时更新 FPS (比在画面上画字快)
        cv2.setWindowTitle("VALORANT Detection", f"FPS: {self.fps:.1f} | Targets: {len(detections)}")
        cv2.imshow("VALORANT Detection", frame)

def main():
    # 配置
    CONFIG = {
        "model_path": "models/val_kenny_ultra_256_v11s.xml",
        "center_size": 256,
        "device": "CPU", # i3-10105F
        "conf_threshold": 0.65, # 进一步提高阈值减少无效检测
        "iou_threshold": 0.35,
        "filter_class": "head"
    }

    app = SimpleRealtimeDetector(**CONFIG)
    app.run()

if __name__ == "__main__":
    main()