"""
VALORANT YOLOv11 视频检测器
支持实时视频处理和录制
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
from yolo_detector import YOLOv11Detector


def detect_video(model_path: str, video_path: str, output_path: str = None,
                conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                show_display: bool = True, save_video: bool = True):
    """
    检测视频

    Args:
        model_path: ONNX模型路径
        video_path: 输入视频路径
        output_path: 输出视频路径
        conf_threshold: 置信度阈值
        iou_threshold: NMS IOU阈值
        show_display: 是否显示实时画面
        save_video: 是否保存视频
    """
    # 创建检测器
    print("\n" + "="*80)
    print("VALORANT YOLOv11 视频检测器")
    print("="*80)

    detector = YOLOv11Detector(model_path, conf_threshold, iou_threshold)

    # 打开视频
    print(f"\n[INFO] 打开视频: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] 视频信息:")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 帧率: {fps} FPS")
    print(f"  - 总帧数: {total_frames}")

    # 创建视频写入器
    video_writer = None
    if save_video:
        if output_path is None:
            output_path = f"output_{Path(video_path).stem}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] 输出视频: {output_path}")

    # 统计信息
    frame_count = 0
    total_inference_time = 0
    total_detections = 0

    print(f"\n[INFO] 开始处理视频...")
    print(f"[INFO] 按 'q' 键退出, 按 'p' 键暂停/继续")

    paused = False
    start_time = time.time()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # 检测
                detections, inference_time = detector.detect(frame)
                total_inference_time += inference_time
                total_detections += len(detections)

                # 绘制结果
                result_frame = detector.draw_detections(frame, detections)

                # 添加信息
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                info_text = f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Objects: {len(detections)}"

                cv2.putText(
                    result_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # 添加进度条
                progress = frame_count / total_frames
                bar_width = width - 40
                bar_height = 20
                bar_x = 20
                bar_y = height - 40

                # 绘制进度条背景
                cv2.rectangle(
                    result_frame,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (50, 50, 50),
                    -1
                )

                # 绘制进度
                progress_width = int(bar_width * progress)
                cv2.rectangle(
                    result_frame,
                    (bar_x, bar_y),
                    (bar_x + progress_width, bar_y + bar_height),
                    (0, 255, 0),
                    -1
                )

                # 进度百分比
                cv2.putText(
                    result_frame,
                    f"{progress*100:.1f}%",
                    (bar_x + bar_width + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                # 保存视频
                if save_video and video_writer is not None:
                    video_writer.write(result_frame)

                # 显示
                if show_display:
                    cv2.imshow("VALORANT Detection", result_frame)

                # 打印进度
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
                    print(f"[PROGRESS] 处理进度: {progress*100:.1f}% "
                          f"({frame_count}/{total_frames}) | "
                          f"平均FPS: {avg_fps:.1f} | "
                          f"预计剩余: {eta:.1f}秒")

            # 按键控制
            if show_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] 用户中断")
                    break
                elif key == ord('p'):
                    paused = not paused
                    status = "暂停" if paused else "继续"
                    print(f"[INFO] {status}")
            else:
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\n[INFO] 检测到Ctrl+C, 正在退出...")

    finally:
        # 清理资源
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if show_display:
            cv2.destroyAllWindows()

        # 输出统计信息
        total_time = time.time() - start_time
        avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print(f"\n" + "="*80)
        print("处理完成!")
        print("="*80)
        print(f"[STATS] 统计信息:")
        print(f"  - 处理帧数: {frame_count}/{total_frames}")
        print(f"  - 总耗时: {total_time:.2f}秒")
        print(f"  - 平均推理时间: {avg_inference_time*1000:.2f}ms")
        print(f"  - 平均FPS: {avg_fps:.2f}")
        print(f"  - 总检测数: {total_detections}")
        print(f"  - 平均每帧检测数: {total_detections/frame_count:.2f}")

        if save_video:
            print(f"\n[INFO] 视频已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VALORANT YOLOv11 视频检测器")

    parser.add_argument(
        "--model",
        type=str,
        default="v11moudle/val_kenny_ultra_256_v11s.onnx",
        help="ONNX模型路径"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="12月24日.mp4",
        help="输入视频路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出视频路径 (默认: output_[输入文件名].mp4)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值 (默认: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IOU阈值 (默认: 0.45)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="不显示实时画面"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存视频"
    )

    args = parser.parse_args()

    # 检测视频
    detect_video(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        show_display=not args.no_display,
        save_video=not args.no_save
    )


if __name__ == "__main__":
    main()
