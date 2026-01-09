# 🚀 快速启动

## 三步运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 转换模型（首次）
python convert_model.py

# 3. 运行检测
python simple_detector.py
```

## 📊 输出

```
距离:  106.4px | 偏移: ( +10.5,  -40.2)
```

- **距离**: 头部到准星的距离（像素）
- **偏移**: X 和 Y 轴偏移（像素）

## ⚙️ 配置（可选）

编辑 `simple_detector.py`:

```python
center_size = 256          # 捕获区域大小
conf_threshold = 0.50      # 置信度阈值
iou_threshold = 0.35       # NMS 阈值
```

## 📝 要求

- Moonlight **全屏**运行
- NVIDIA GPU（支持 CUDA）
- CUDA >= 11.8
- TensorRT >= 8.6

## ⚡ 预期性能

- RTX 3060+: 200+ FPS
- RTX 4060+: 300+ FPS

详细文档见 `README.md`
