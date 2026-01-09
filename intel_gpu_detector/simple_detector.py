"""
VALORANT 实时检测系统 - 精简版
仅输出头部到屏幕中心的距离
"""
import cv2
import numpy as np
import dxcam
import time
from collections import deque
from typing import List, Tuple, Dict, Optional
from yolo_detector_openvino import YOLOv11DetectorOpenVINO


class SimpleRealtimeDetector:
    """极简实时检测器"""

    def __init__(self,
                 model_path: str,
                 center_size: int = 640,
                 device: str = "GPU",
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 filter_class: Optional[str] = None):
        """
        初始化检测器

        Args:
            model_path: OpenVINO 模型路径
            center_size: 中心捕获区域大小
            device: 推理设备 (GPU/CPU)
            conf_threshold: 置信度阈值
            iou_threshold: NMS 阈值
            filter_class: 只保留指定类别 (如 "head")
        """
        print("="*80)
        print(f"VALORANT 实时检测系统 - 精简版 (捕获中心 {center_size}x{center_size})")
        print("="*80)

        self.center_size = center_size

        # 加载检测模型
        print("\n[1/2] 加载 OpenVINO 检测模型...")
        self.detector = YOLOv11DetectorOpenVINO(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            filter_class=filter_class
        )

        # 计算捕获区域
        print("\n[2/2] 计算捕获区域...")
        self.capture_region = self._get_center_region()
        print(f"[INFO] 捕获区域: {self.capture_region}")

        self.camera = None

        # FPS 计算（使用滑动窗口，更准确）
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()

        print("\n[SUCCESS] 系统初始化完成!")
        print("="*80)

    def _get_center_region(self) -> Tuple[int, int, int, int]:
        """计算屏幕中心区域坐标"""
        camera_temp = dxcam.create()
        frame = camera_temp.grab()

        if frame is None:
            screen_width, screen_height = 1920, 1080
            print(f"[WARNING] 无法获取屏幕尺寸，使用默认 {screen_width}x{screen_height}")
        else:
            screen_height, screen_width = frame.shape[:2]
            print(f"[INFO] 检测到屏幕尺寸: {screen_width} x {screen_height}")

        del camera_temp

        # 计算中心区域
        center_x = screen_width // 2
        center_y = screen_height // 2
        half_size = self.center_size // 2

        left = center_x - half_size
        top = center_y - half_size
        right = center_x + half_size
        bottom = center_y + half_size

        return (left, top, right, bottom)

    def find_closest_head(self, detections: List[Dict], screen_center: Tuple[int, int]) -> Optional[Dict]:
        """找到距离中心最近的 head"""
        heads = [det for det in detections if det['class_name'] == 'head']

        if not heads:
            return None

        closest_head = None
        min_distance_sq = float('inf')

        for head in heads:
            # 目标中心点
            x1, y1, x2, y2 = head['bbox']
            cx, cy = screen_center
            target_x = (x1 + x2) * 0.5
            target_y = (y1 + y2) * 0.5

            # 偏移量
            offset_x = target_x - cx
            offset_y = target_y - cy

            # 距离平方（用于比较）
            distance_sq = offset_x * offset_x + offset_y * offset_y

            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest_head = {
                    **head,
                    'distance': distance_sq ** 0.5,
                    'offset_x': offset_x,
                    'offset_y': offset_y
                }

        return closest_head

    def run(self):
        """运行实时检测"""
        print("\n" + "="*80)
        print("开始实时检测 - 按 'Q' 退出")
        print("="*80 + "\n")

        # 创建屏幕捕获
        self.camera = dxcam.create(output_color="BGR")
        self.camera.start(target_fps=0, region=self.capture_region)

        try:
            while True:
                loop_start = time.perf_counter()

                # 捕获帧
                frame = self.camera.get_latest_frame()
                if frame is None:
                    continue

                # 计算屏幕中心
                height, width = frame.shape[:2]
                screen_center = (width // 2, height // 2)

                # 检测
                detections, _, _ = self.detector.detect(frame)

                # 找到最近的 head
                closest_head = self.find_closest_head(detections, screen_center)

                # 可视化
                vis_frame = self._create_visualization(
                    frame, detections, closest_head, screen_center, height
                )
                cv2.imshow("VALORANT Detection", vis_frame)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                # 计算总时间
                loop_end = time.perf_counter()
                total_time = loop_end - loop_start

                # 更新 FPS
                self.frame_times.append(total_time)
                avg_time = sum(self.frame_times) / len(self.frame_times)
                self.fps = int(1.0 / avg_time) if avg_time > 0 else 0

        except KeyboardInterrupt:
            print("\n\n检测已停止")

        finally:
            if self.camera:
                self.camera.stop()
            cv2.destroyAllWindows()

            print("\n" + "="*80)
            print("系统已关闭")
            print("="*80)

    def _create_visualization(self, frame: np.ndarray, detections: List[Dict],
                             closest_head: Optional[Dict], screen_center: Tuple[int, int],
                             height: int) -> np.ndarray:
        """创建可视化帧"""
        vis_frame = frame.copy()

        # 1. 绘制中心点
        cv2.circle(vis_frame, screen_center, 5, (0, 255, 0), -1)
        cv2.circle(vis_frame, screen_center, 20, (0, 255, 0), 2)

        # 2. 绘制所有 head
        heads = [det for det in detections if det['class_name'] == 'head']
        for head in heads:
            x1, y1, x2, y2 = map(int, head['bbox'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_frame, "HEAD", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 3. 绘制最近的 head 和距离
        if closest_head:
            x1, y1, x2, y2 = map(int, closest_head['bbox'])
            target_x = int((x1 + x2) / 2)
            target_y = int((y1 + y2) / 2)

            # 绘制从中心点到目标的线
            cv2.line(vis_frame, screen_center, (target_x, target_y), (255, 0, 0), 2)

            # 显示距离和偏移
            dist_text = f"Dist: {closest_head['distance']:.1f}px"
            offset_text = f"Offset: ({closest_head['offset_x']:+.0f}, {closest_head['offset_y']:+.0f})"
            cv2.putText(vis_frame, dist_text, (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_frame, offset_text, (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4. 显示 FPS 和检测数
        fps_text = f"FPS: {self.fps}"
        detect_text = f"Heads: {len(heads)}"
        cv2.putText(vis_frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_frame, detect_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return vis_frame

    def __del__(self):
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()


def main():
    """主函数"""

    # ==================== 配置区 ====================
    #
    # 模型配置
    model_path = "../models/val_kenny_ultra_256_v11s.xml"
    center_size = 256

    # 硬件配置
    device = "CPU"

    # 检测配置（降低阈值减少后处理计算）
    conf_threshold = 0.60  # 从 0.55 提高到 0.60，减少检测数量
    iou_threshold = 0.40   # 从 0.35 提高到 0.40，减少 NMS 计算
    filter_class = "head"  # 只检测 head 类别，大幅提升性能

    # ================================================

    print("="*80)
    print("[INFO] 配置:")
    print(f"  - 硬件: Intel i3-10105F")
    print(f"  - 模型: 256x256")
    print(f"  - 设备: {device}")
    print("="*80)
    print()

    # 创建检测器
    detector = SimpleRealtimeDetector(
        model_path=model_path,
        center_size=center_size,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        filter_class=filter_class
    )

    # 运行检测
    detector.run()


if __name__ == "__main__":
    main()
