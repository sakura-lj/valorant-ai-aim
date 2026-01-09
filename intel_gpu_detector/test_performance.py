"""
VALORANT 实时检测系统 - 极简无显示版本
用于测试纯推理性能
"""
import numpy as np
import dxcam
import time
from typing import List, Tuple, Dict, Optional, Union
from yolo_detector_openvino import YOLOv11DetectorOpenVINO


class MinimalDetector:
    """极简检测器 - 无可视化，纯性能测试"""

    def __init__(self, model_path: str, center_size: int = 256,
                 device: str = "CPU", conf_threshold: float = 0.55,
                 iou_threshold: float = 0.35, filter_class: Optional[str] = None):
        print("="*80)
        print(f"VALORANT 极简检测器 - 纯性能测试")
        print("="*80)

        self.center_size = center_size

        # 加载检测模型
        print("\n[1/2] 加载模型...")
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
        self.fps = 0
        self.frame_count = 0
        self.start_time = None

        print("\n[SUCCESS] 初始化完成!")
        print("="*80)

    def _get_center_region(self) -> Tuple[int, int, int, int]:
        """计算屏幕中心区域坐标"""
        camera_temp = dxcam.create()
        frame = camera_temp.grab()

        if frame is None:
            screen_width, screen_height = 1920, 1080
        else:
            screen_height, screen_width = frame.shape[:2]

        del camera_temp

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

        min_distance_sq = float('inf')
        closest_head = None

        cx, cy = screen_center
        for head in heads:
            x1, y1, x2, y2 = head['bbox']
            target_x = (x1 + x2) * 0.5
            target_y = (y1 + y2) * 0.5

            offset_x = target_x - cx
            offset_y = target_y - cy
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

    def run(self, duration: int = 30):
        """运行检测并测试性能

        Args:
            duration: 测试持续时间（秒）
        """
        print("\n" + "="*80)
        print(f"开始性能测试 - 运行 {duration} 秒（无可视化，纯推理）")
        print("按 Ctrl+C 可提前退出")
        print("="*80 + "\n")

        # 创建屏幕捕获
        self.camera = dxcam.create(output_color="BGR")
        self.camera.start(target_fps=0, region=self.capture_region)

        self.frame_count = 0
        self.start_time = time.time()
        last_print_time = self.start_time

        try:
            while True:
                # 检查是否超时
                elapsed = time.time() - self.start_time
                if elapsed >= duration:
                    break

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

                # 更新帧计数
                self.frame_count += 1

                # 每秒打印一次统计
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.start_time)
                    heads_count = len([d for d in detections if d['class_name'] == 'head'])

                    if closest_head:
                        print(f"[{elapsed:5.1f}s] FPS: {self.fps:5.1f} | Heads: {heads_count} | "
                              f"Closest: {closest_head['distance']:5.1f}px")
                    else:
                        print(f"[{elapsed:5.1f}s] FPS: {self.fps:5.1f} | Heads: {heads_count} | "
                              f"Closest: None")

                    last_print_time = current_time

        except KeyboardInterrupt:
            print("\n\n测试中断")

        finally:
            if self.camera:
                self.camera.stop()

            # 最终统计
            total_time = time.time() - self.start_time
            final_fps = self.frame_count / total_time if total_time > 0 else 0
            avg_latency = (total_time / self.frame_count * 1000) if self.frame_count > 0 else 0

            print("\n" + "="*80)
            print("性能测试结果")
            print("="*80)
            print(f"总帧数: {self.frame_count}")
            print(f"总时长: {total_time:.2f}秒")
            print(f"平均 FPS: {final_fps:.2f}")
            print(f"平均延迟: {avg_latency:.2f}ms")
            print("="*80)

    def __del__(self):
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()


if __name__ == "__main__":
    print("\n配置说明:")
    print("  - conf_threshold: 0.60 (提高阈值减少检测)")
    print("  - iou_threshold: 0.40")
    print("  - filter_class: 'head' (只检测头部，大幅提升性能)")
    print()

    detector = MinimalDetector(
        model_path="../models/val_kenny_ultra_256_v11s.xml",
        center_size=256,
        device="CPU",
        conf_threshold=0.60,  # 提高阈值
        iou_threshold=0.40,
        filter_class="head"   # 只检测 head 类别
    )

    # 运行 30 秒性能测试
    detector.run(duration=30)
