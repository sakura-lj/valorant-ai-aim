# VALORANT YOLOv11 物体检测系统

基于YOLOv11的无畏契约游戏物体识别程序，支持检测头部、敌人、队友、道具和闪光。

## 检测类别

- 0: 头
- 1: 敌人
- 2: 队友
- 3: 道具
- 4: 闪光

## 文件说明

### 核心文件
- `yolo_detector.py` - 核心检测器类和图像检测脚本
- `video_detector.py` - 视频检测脚本
- `analyze_model.py` - ONNX模型分析工具

### 模型文件
`v11moudle/` 目录包含4个不同尺寸的模型:
- `val_kenny_ultra_256_v11s.onnx` - 256x256 (最快)
- `val_kenny_ultra_320_v11s.onnx` - 320x320
- `kenny_ultra_416_v11s.onnx` - 416x416
- `kenny_ultra_640_v11s.onnx` - 640x640 (最准确)

## 依赖安装

```bash
pip install onnxruntime opencv-python numpy onnx
```

## 使用方法

### 1. 分析模型结构

查看所有模型的输入输出信息:

```bash
python analyze_model.py
```

### 2. 图像检测

检测单张图片:

```bash
python yolo_detector.py
```

默认配置:
- 模型: `kenny_ultra_640_v11s.onnx`
- 输入: `test.png`
- 输出: `result.png`
- 置信度阈值: 0.25
- NMS阈值: 0.45

### 3. 视频检测

检测视频文件:

```bash
python video_detector.py
```

#### 命令行参数

```bash
# 基本用法
python video_detector.py --video 12月24日.mp4

# 指定模型和输出路径
python video_detector.py --model v11moudle/kenny_ultra_416_v11s.onnx --video input.mp4 --output output.mp4

# 调整检测阈值
python video_detector.py --conf 0.3 --iou 0.5

# 不显示实时画面 (加快处理速度)
python video_detector.py --no-display

# 只显示不保存
python video_detector.py --no-save
```

#### 视频控制键

- `q` - 退出
- `p` - 暂停/继续

## 自定义使用

### 在代码中使用检测器

```python
from yolo_detector import YOLOv11Detector
import cv2

# 创建检测器
detector = YOLOv11Detector(
    model_path="v11moudle/kenny_ultra_640_v11s.onnx",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# 读取图像
image = cv2.imread("test.png")

# 检测
detections, inference_time = detector.detect(image)

# 处理结果
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
    print(f"位置: {det['bbox']}")

# 绘制结果
result_image = detector.draw_detections(image, detections)
cv2.imwrite("output.png", result_image)
```

## 模型选择建议

### 实时检测 (追求速度)
推荐使用 `val_kenny_ultra_256_v11s.onnx` 或 `val_kenny_ultra_320_v11s.onnx`

### 录像分析 (追求准确率)
推荐使用 `kenny_ultra_640_v11s.onnx`

### 平衡选择
推荐使用 `kenny_ultra_416_v11s.onnx`

## 性能参考

在CPU (Intel Core i5) 上的推理时间:
- 256x256: ~50ms (20 FPS)
- 320x320: ~70ms (14 FPS)
- 416x416: ~110ms (9 FPS)
- 640x640: ~145ms (7 FPS)

*实际性能取决于CPU性能*

## 检测结果

检测结果为字典列表，每个检测包含:

```python
{
    'bbox': [x1, y1, x2, y2],  # 边界框坐标 (左上角和右下角)
    'confidence': 0.85,         # 置信度分数
    'class_id': 1,              # 类别ID
    'class_name': '敌人'        # 类别名称
}
```

## 颜色编码

不同类别使用不同颜色标注:
- 头部: 红色
- 敌人: 深红色
- 队友: 绿色
- 道具: 青色
- 闪光: 橙色

## 常见问题

### 检测不到目标?
1. 降低置信度阈值: `--conf 0.15`
2. 尝试更大的模型: `--model v11moudle/kenny_ultra_640_v11s.onnx`

### 检测速度太慢?
1. 使用更小的模型: `--model v11moudle/val_kenny_ultra_256_v11s.onnx`
2. 关闭实时显示: `--no-display`

### 误检太多?
1. 提高置信度阈值: `--conf 0.35`
2. 调整NMS阈值: `--iou 0.3`

## 技术细节

### 预处理
1. 保持宽高比缩放到模型输入尺寸
2. 居中填充 (灰色背景)
3. BGR转RGB
4. 归一化到 [0, 1]
5. HWC转CHW格式

### 后处理
1. 置信度过滤
2. 坐标转换 (中心点 -> 左上角)
3. 去除padding并缩放回原始尺寸
4. 按类别进行NMS (非极大值抑制)

## 项目结构

```
VALORANT/
├── v11moudle/                  # 模型文件夹
│   ├── val_kenny_ultra_256_v11s.onnx
│   ├── val_kenny_ultra_320_v11s.onnx
│   ├── kenny_ultra_416_v11s.onnx
│   └── kenny_ultra_640_v11s.onnx
├── analyze_model.py            # 模型分析工具
├── yolo_detector.py            # 核心检测器
├── video_detector.py           # 视频检测器
├── test.png                    # 测试图片
├── 12月24日.mp4                # 测试视频
└── README.md                   # 本文档
```

## 许可

本项目仅供学习和研究使用。
