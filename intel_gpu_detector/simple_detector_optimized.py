"""
VALORANT 实时检测系统 - CPU 极限优化版
专为低端 CPU 优化：i3-10105F + GT 1030（不使用 GPU）
目标：60+ FPS

优化策略：
1. 关闭所有可视化（最大性能提升）
2. 使用 224x224 小模型
3. 只检测 head 类别
4. 高置信度阈值
5. 纯控制台输出
"""
import numpy as np
import dxcam
import time
from collections import deque
from typing import List, Tuple, Dict, Optional
from yolo_detector_openvino import YOLOv11DetectorOpenVINO


class OptimizedRealtimeDetector:
    """CPU 极限优化检测器 - 无可视化版本"""

    def __init__(self,
                 model_path: str,
                 center_size: int = 224,
                 device: str = "CPU",
                 conf_threshold: float = 0.65,
                 iou_threshold: float = 0.45):
        """
        初始化检测器

        Args:
            model_path: OpenVINO 模型路径
            center_size: 中心捕获区域大小（推荐 224）
            device: 推理设备（CPU）
            conf_threshold: 置信度阈值（越高越快）
            iou_threshold: NMS 阈值
        """
        print("="*80)
        print(f"VALORANT CPU 极限优化版 (捕获中心 {center_size}x{center_size})")
        print("="*80)

        self.center_size = center_size

        # 加载检测模型 - 只检测 head 类别
        print("\n[1/2] 加载 OpenVINO 检测模型...")
        self.detector = YOLOv11DetectorOpenVINO(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            filter_class="head"  # 只检测头部，大幅提升性能
        )

        # 计算捕获区域
        print("\n[2/2] 计算捕获区域...")
        self.capture_region = self._get_center_region()
        print(f"[INFO] 捕获区域: {self.capture_region}")

        self.camera = None

        # FPS 计算
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        self.frame_count = 0
        self.last_print_time = time.time()

        print("\n[SUCCESS] 系统初始化完成!")
        print("\n优化设置:")
        print(f"  - 捕获区域: {center_size}x{center_size}")
        print(f"  - 置信度阈值: {conf_threshold}")
        print(f"  - 可视化: 关闭 (性能模式)")
        print(f"  - 只检测: head 类别")
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
        """找到距离中心最近的 head（向量化优化版本）"""
        if not detections:
            return None

        # 向量化计算所有距离
        centers_x = np.array([(d['bbox'][0] + d['bbox'][2]) * 0.5 for d in detections])
        centers_y = np.array([(d['bbox'][1] + d['bbox'][3]) * 0.5 for d in detections])

        cx, cy = screen_center
        offset_x = centers_x - cx
        offset_y = centers_y - cy

        distances_sq = offset_x * offset_x + offset_y * offset_y
        min_idx = distances_sq.argmin()

        return {
            **detections[min_idx],
            'distance': float(distances_sq[min_idx] ** 0.5),
            'offset_x': float(offset_x[min_idx]),
            'offset_y': float(offset_y[min_idx])
        }

    def run(self):
        """运行实时检测 - 无可视化版本"""
        print("\n" + "="*80)
        print("开始实时检测 - 按 Ctrl+C 退出")
        print("输出格式: [FPS] 距离 | 偏移")
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

                # 计算 FPS
                loop_end = time.perf_counter()
                total_time = loop_end - loop_start
                self.frame_times.append(total_time)
                avg_time = sum(self.frame_times) / len(self.frame_times)
                self.fps = int(1.0 / avg_time) if avg_time > 0 else 0

                # 每秒输出一次信息（减少 I/O）
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_print_time >= 0.1:  # 每 100ms 更新一次
                    if closest_head:
                        print(f"\r[{self.fps:3d} FPS] 距离: {closest_head['distance']:6.1f}px | "
                              f"偏移: ({closest_head['offset_x']:+6.1f}, {closest_head['offset_y']:+6.1f})",
                              end='', flush=True)
                    else:
                        print(f"\r[{self.fps:3d} FPS] 未检测到目标", end='', flush=True)

                    self.last_print_time = current_time

        except KeyboardInterrupt:
            print("\n\n检测已停止")

        finally:
            if self.camera:
                self.camera.stop()

            print("\n" + "="*80)
            print(f"平均 FPS: {self.fps}")
            print("系统已关闭")
            print("="*80)

    def __del__(self):
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()


def main():
    """主函数 - CPU 极限优化配置"""

    # ==================== 配置区 ====================

    # 模型配置（使用最小模型）
    model_path = "models/val_kenny_ultra_256_v11s.xml"
    center_size = 224  # 减小到 224 提升性能

    # 硬件配置
    device = "CPU"

    # 检测配置（高阈值 = 更快）
    conf_threshold = 0.65  # 高阈值，减少误检和后处理
    iou_threshold = 0.45

    # ================================================

    print("="*80)
    print("[INFO] CPU 极限优化配置:")
    print(f"  - CPU: Intel i3-10105F")
    print(f"  - 模型: 256x256 (输入) -> 224x224 (捕获)")
    print(f"  - 可视化: 关闭")
    print(f"  - 置信度: {conf_threshold}")
    print(f"  - 预期 FPS: 50-70")
    print("="*80)
    print()

    # 创建检测器
    detector = OptimizedRealtimeDetector(
        model_path=model_path,
        center_size=center_size,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

    # 运行检测
    detector.run()


if __name__ == "__main__":
    main()
