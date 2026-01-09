# 🚀 快速启动

## 一步运行（无需转换模型）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 直接运行检测
python simple_detector.py
```

**就是这么简单！无需任何模型转换！**

---

## 📊 输出

```
距离:  106.4px | 偏移: ( +10.5,  -40.2)
```

- **距离**: 头部到准星的距离（像素）
- **偏移**: X 和 Y 轴偏移（像素）

---

## ⚙️ 配置（可选）

编辑 `simple_detector.py`:

```python
model_path = "../v11moudle/kenny_ultra_640_v11s.onnx"  # ONNX 模型
center_size = 640          # 捕获区域大小
conf_threshold = 0.50      # 置信度阈值
iou_threshold = 0.35       # NMS 阈值
```

---

## 📝 要求

- Moonlight **全屏**运行
- NVIDIA GPU（任何型号）
- CUDA 驱动已安装

---

## ⚡ 预期性能

- GTX 1650+: 80-100 FPS
- RTX 3060+: 150-180 FPS
- RTX 4060+: 200+ FPS

---

## 💡 优势

- ✅ **无需模型转换** - 直接使用 ONNX 模型
- ✅ **简单安装** - 只需 pip install
- ✅ **所有 GPU** - 支持所有 NVIDIA 显卡
- ✅ **高性能** - CUDA 加速推理

详细文档见 [README.md](README.md)
