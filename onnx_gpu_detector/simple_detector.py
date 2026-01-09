"""
VALORANT 实时检测系统 - ONNX Runtime GPU 版本
仅输出头部到屏幕中心的距离
"""
import cv2
import numpy as np
import dxcam
import time
from typing import List, Tuple, Dict, Optional
from yolo_detector_onnx import YOLOv11DetectorONNX


class SimpleRealtimeDetector:
    """极简实时检测器 - ONNX Runtime GPU 版本"""

    def __init__(self,
                 model_path: str,
                 center_size: int = 640,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 enable_profiling: bool = True):
        """
        初始化检测器

        Args:
            model_path: ONNX 模型路径
            center_size: 中心捕获区域大小
            conf_threshold: 置信度阈值
            iou_threshold: NMS 阈值
            enable_profiling: 是否启用性能分析
        """
        print("="*80)
        print(f"VALORANT 实时检测系统 - ONNX Runtime GPU 版本 (捕获中心 {center_size}x{center_size})")
        print("="*80)

        self.center_size = center_size
        self.enable_profiling = enable_profiling

        # 加载检测模型
        print("\n[1/2] 加载 ONNX 检测模型...")
        self.detector = YOLOv11DetectorONNX(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            enable_profiling=enable_profiling
        )

        # 计算捕获区域
        print("\n[2/2] 计算捕获区域...")
        self.capture_region = self._get_center_region()
        print(f"[INFO] 捕获区域: {self.capture_region}")

        self.camera = None

        # FPS 计算（使用滑动窗口，更准确）
        self.fps = 0
        self.frame_times = []
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
        print("开始实时检测")
        print("  按 'Q' 退出")
        if self.enable_profiling:
            print("  按 'P' 显示性能统计")
        print("="*80 + "\n")

        # 创建屏幕捕获
        self.camera = dxcam.create(output_color="BGR")
        self.camera.start(target_fps=0, region=self.capture_region)

        try:
            while True:
                # 捕获帧
                frame = self.camera.get_latest_frame()
                if frame is None:
                    continue

                # 计算屏幕中心
                height, width = frame.shape[:2]
                screen_center = (width // 2, height // 2)

                # 检测（计时开始）
                detect_start = time.time()
                detections, _, _ = self.detector.detect(frame)

                # 找到最近的 head
                closest_head = self.find_closest_head(detections, screen_center)
                detect_end = time.time()

                # 更新 FPS（只计算检测时间，不包含显示）
                detect_time = detect_end - detect_start
                self.frame_times.append(detect_time)
                if len(self.frame_times) > 30:  # 保留最近 30 帧
                    self.frame_times.pop(0)
                avg_time = sum(self.frame_times) / len(self.frame_times)
                self.fps = int(1.0 / avg_time) if avg_time > 0 else 0

                # 可视化
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

                    # 显示距离
                    dist_text = f"Distance: {closest_head['distance']:.1f}px"
                    cv2.putText(vis_frame, dist_text, (10, height - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 4. 显示 FPS
                fps_text = f"FPS: {self.fps}"
                cv2.putText(vis_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 显示
                cv2.imshow("VALORANT Detection - ONNX Runtime GPU", vis_frame)

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p') and self.enable_profiling:
                    # 显示性能统计
                    self.detector.print_performance_stats()


        except KeyboardInterrupt:
            print("\n\n检测已停止")

        finally:
            if self.camera:
                self.camera.stop()
            cv2.destroyAllWindows()

            # 最终性能报告
            if self.enable_profiling:
                print("\n最终性能报告:")
                self.detector.print_performance_stats()

            print("\n" + "="*80)
            print("系统已关闭")
            print("="*80)

    def __del__(self):
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()


def main():
    """主函数"""

    # 配置
    model_path = "../v11moudle/kenny_ultra_640_v11s.onnx"  # ONNX 模型路径
    center_size = 640
    conf_threshold = 0.50
    iou_threshold = 0.35
    enable_profiling = True  # 启用性能分析

    # 创建检测器
    detector = SimpleRealtimeDetector(
        model_path=model_path,
        center_size=center_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        enable_profiling=enable_profiling
    )

    # 运行检测
    detector.run()


if __name__ == "__main__":
    main()
