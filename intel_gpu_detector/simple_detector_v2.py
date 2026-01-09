"""
VALORANT 实时检测系统 - CPU 深度优化版
针对 i3-10105F + GT 1030 优化
核心优化：OpenVINO PPP + 纯 NumPy 流水线 + 轻量级渲染
"""
import cv2
import numpy as np
import dxcam
import time
from collections import deque
from typing import Tuple, Optional, Dict
from yolo_detector_openvino_v2 import YOLOv11DetectorOpenVINO


class OptimizedRealtimeDetector:
    """深度优化的实时检测器"""

    def __init__(self,
                 model_path: str,
                 center_size: int = 256,
                 device: str = "CPU",
                 conf_threshold: float = 0.60,
                 iou_threshold: float = 0.40,
                 filter_class: str = "head"):
        """
        初始化检测器

        优化点：
        1. 预计算屏幕中心坐标
        2. 预启动相机（target_fps=200）
        3. 使用优化后的检测器（OpenVINO PPP）
        """
        print("="*80)
        print(f"VALORANT 深度优化版 (捕获 {center_size}x{center_size})")
        print("="*80)

        self.center_size = center_size
        # ============ 优化 1: 预计算屏幕中心 ============
        self.screen_cx = center_size // 2
        self.screen_cy = center_size // 2

        # ============ 优化 2: 使用 OpenVINO PPP 优化的检测器 ============
        print("\n[1/3] 加载 OpenVINO 推理引擎 (PPP 优化版)...")
        self.detector = YOLOv11DetectorOpenVINO(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            filter_class=filter_class
        )

        # ============ 优化 3: 计算捕获区域并预启动相机 ============
        print("\n[2/3] 初始化 DXCam 捕获...")
        self.capture_region = self._get_center_region()
        print(f"[INFO] 捕获区域: {self.capture_region}")

        print("\n[3/3] 预启动相机...")
        self.camera = dxcam.create(output_color="BGR")
        # 预先启动，设置高 target_fps 避免等待
        self.camera.start(region=self.capture_region, target_fps=200)

        # FPS 统计
        self.frame_times = deque(maxlen=30)
        self.fps = 0

        print("\n[SUCCESS] 系统初始化完成!")
        print("="*80)

    def _get_center_region(self) -> Tuple[int, int, int, int]:
        """
        计算屏幕中心区域坐标

        优化：使用 dxcam 对象的属性获取屏幕尺寸（更快）
        备用：如果没有属性，则 grab() 一帧
        """
        tmp = dxcam.create()

        # 尝试使用属性（更快）
        if hasattr(tmp, 'width') and hasattr(tmp, 'height'):
            screen_width, screen_height = tmp.width, tmp.height
            print(f"[INFO] 屏幕尺寸: {screen_width} x {screen_height}")
        else:
            # 备用方案：抓取一帧
            frame = tmp.grab()
            if frame is not None:
                screen_height, screen_width = frame.shape[:2]
                print(f"[INFO] 屏幕尺寸: {screen_width} x {screen_height}")
            else:
                screen_width, screen_height = 1920, 1080
                print(f"[WARNING] 无法获取屏幕尺寸，使用默认 {screen_width}x{screen_height}")

        del tmp

        # 计算中心区域
        left = (screen_width - self.center_size) // 2
        top = (screen_height - self.center_size) // 2
        right = left + self.center_size
        bottom = top + self.center_size

        return (left, top, right, bottom)

    def find_best_target(self, detections: np.ndarray) -> Optional[Dict]:
        """
        ============ 优化 4: 纯 NumPy 向量化寻找最近目标 ============
        使用向量化操作，避免 Python 循环

        Args:
            detections: NumPy 数组 [[x1, y1, x2, y2, conf, cls], ...]

        Returns:
            最近目标的信息字典，或 None
        """
        if len(detections) == 0:
            return None

        # 1. 向量化计算所有目标的中心坐标
        # centers shape: [N, 2] = [[cx1, cy1], [cx2, cy2], ...]
        centers = (detections[:, :2] + detections[:, 2:4]) * 0.5

        # 2. 向量化计算到屏幕中心的欧氏距离
        # 使用广播: centers - [screen_cx, screen_cy]
        diffs = centers - np.array([self.screen_cx, self.screen_cy])
        distances = np.linalg.norm(diffs, axis=1)

        # 3. 找到最小距离的索引
        idx = np.argmin(distances)

        # 4. 返回结果（轻量级字典）
        return {
            'bbox': detections[idx, :4],
            'conf': float(detections[idx, 4]),
            'dist': float(distances[idx]),
            'ox': float(diffs[idx, 0]),
            'oy': float(diffs[idx, 1])
        }

    def run(self):
        """
        主循环：抓图 -> 检测 -> 渲染

        优化点：
        - 使用 perf_counter 精确计时
        - 只在有检测时才绘制
        - FPS 显示在标题栏
        """
        print("\n" + "="*80)
        print("检测运行中... 按 'Q' 键退出")
        print("="*80 + "\n")

        try:
            while True:
                t1 = time.perf_counter()

                # 1. 快速抓图
                frame = self.camera.get_latest_frame()
                if frame is None:
                    continue

                # 2. 推理（返回 NumPy 数组）
                detections = self.detector.detect(frame)

                # 3. 寻找最近目标（纯 NumPy）
                best_target = self.find_best_target(detections)

                # 4. 轻量级可视化
                self._render(frame, detections, best_target)

                # 5. FPS 统计
                self.frame_times.append(time.perf_counter() - t1)
                if len(self.frame_times) >= 30:
                    avg_time = sum(self.frame_times) / 30
                    self.fps = 1.0 / avg_time if avg_time > 0 else 0

                # 6. 处理按键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n\n检测已停止")

        finally:
            if self.camera:
                self.camera.stop()
            cv2.destroyAllWindows()

            print("\n" + "="*80)
            print(f"平均 FPS: {self.fps:.1f}")
            print("系统已关闭")
            print("="*80)

    def _render(self, frame: np.ndarray, detections: np.ndarray, target: Optional[Dict]):
        """
        ============ 优化 5: 轻量级可视化 ============
        减少绘制开销：
        - 使用 1px 线宽
        - FPS 显示在标题栏而非画面
        - 使用 drawMarker 代替 circle
        """
        # 绘制中心准星
        cv2.drawMarker(frame, (self.screen_cx, self.screen_cy),
                      (0, 255, 0), cv2.MARKER_CROSS, 15, 1)

        # 绘制所有检测框（1px 线宽）
        for det in detections:
            x1, y1, x2, y2 = det[:4].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # 绘制锁定目标
        if target:
            bbox = target['bbox']
            tx = int((bbox[0] + bbox[2]) / 2)
            ty = int((bbox[1] + bbox[3]) / 2)

            # 绘制连线（1px）
            cv2.line(frame, (self.screen_cx, self.screen_cy), (tx, ty), (255, 255, 0), 1)

            # 显示距离和偏移（小字体）
            info_text = f"DIST: {target['dist']:.1f}px | OFFSET: ({target['ox']:+.0f}, {target['oy']:+.0f})"
            cv2.putText(frame, info_text, (10, self.center_size - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ============ 优化 6: 标题栏显示 FPS ============
        # 比在画面上绘制文字快
        title = f"VALORANT Detection | FPS: {self.fps:.1f} | Targets: {len(detections)}"
        cv2.setWindowTitle("VALORANT Detection", title)

        cv2.imshow("VALORANT Detection", frame)

    def __del__(self):
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()


def main():
    """
    主函数 - i3-10105F CPU 优化配置
    """
    # ==================== 配置区 ====================
    CONFIG = {
        "model_path": "models/val_kenny_ultra_256_v11s.xml",
        "center_size": 256,  # 捕获区域（可降到 224 进一步提速）
        "device": "CPU",  # i3-10105F（无集成显卡）
        "conf_threshold": 0.65,  # 高阈值减少后处理
        "iou_threshold": 0.35,
        "filter_class": "head"  # 只检测头部
    }
    # ================================================

    print("="*80)
    print("[INFO] 深度优化配置:")
    print(f"  - 硬件: Intel i3-10105F (4核CPU)")
    print(f"  - 模型: 256x256")
    print(f"  - 捕获区域: {CONFIG['center_size']}x{CONFIG['center_size']}")
    print(f"  - 置信度: {CONFIG['conf_threshold']}")
    print(f"  - 优化: OpenVINO PPP + 纯NumPy + 线程绑定")
    print(f"  - 预期 FPS: 60-80")
    print("="*80)
    print()

    # 创建检测器
    app = OptimizedRealtimeDetector(**CONFIG)

    # 运行检测
    app.run()


if __name__ == "__main__":
    main()
