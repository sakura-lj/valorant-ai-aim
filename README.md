# VALORANT Real-Time Object Detection System

A high-performance YOLOv11-based object detection system for VALORANT, optimized for different hardware platforms. Designed for real-time head detection through Moonlight game streaming with ultra-low latency.

## Features

- **Real-time Detection**: Detects game objects (enemies, heads, teammates, items, flash) with high accuracy
- **Multiple GPU Backends**: TensorRT, ONNX Runtime GPU, and OpenVINO support
- **Ultra-Low Latency**: As low as 2-5ms total latency on high-end NVIDIA GPUs
- **Optimized Performance**: 200-400 FPS on NVIDIA GPUs, 55-60 FPS on Intel iGPUs
- **Center Region Capture**: Focuses on crosshair area for maximum efficiency
- **Distance Calculation**: Computes distance and offset from detected heads to screen center
- **Moonlight Integration**: Designed for full-screen game streaming setups

## Version Overview

This project includes **4 optimized versions** for different hardware:

| Version | Hardware | FPS | Latency | Best For |
|---------|----------|-----|---------|----------|
| **nvidia_gpu_detector/** | NVIDIA dedicated GPU | 200-400 | ~2-5ms | Maximum performance |
| **onnx_gpu_detector/** | NVIDIA any GPU | 80-250 | ~5-10ms | Easy setup, good performance |
| **intel_gpu_detector/** | Intel integrated GPU | 55-60 | ~17ms | Intel laptops/desktop iGPUs |
| **old_version/** | CPU only | 20-40 | ~35ms+ | Testing and development |

### Quick Selection Guide

- **Have NVIDIA GPU + want maximum FPS** → Use `nvidia_gpu_detector/` (TensorRT)
- **Have NVIDIA GPU + want easy setup** → Use `onnx_gpu_detector/` (no model conversion)
- **Have Intel integrated GPU** → Use `intel_gpu_detector/` (OpenVINO)
- **Just testing/developing** → Use `old_version/` (CPU-based)

## Detection Classes

- **0**: enemy (敌人) - Enemy players
- **1**: head (头部) - Head/headshot target
- **2**: teammate (队友) - Friendly players
- **3**: item (道具) - In-game items
- **4**: flash (闪光) - Flash effects

## Quick Start

### NVIDIA GPU - TensorRT Version (Fastest)

```bash
cd nvidia_gpu_detector

# Install dependencies
pip install -r requirements.txt

# Prerequisites: Install CUDA >= 11.8 and TensorRT >= 8.6 separately

# Convert ONNX to TensorRT engine (hardware-specific, run on target machine)
python convert_model.py

# Run detection
python simple_detector.py
```

**Performance**: 200-400 FPS | ~2-5ms latency | Best for high-end NVIDIA GPUs (RTX 3060+)

### NVIDIA GPU - ONNX Runtime Version (Easiest)

```bash
cd onnx_gpu_detector

# Install dependencies
pip install -r requirements.txt

# Prerequisite: CUDA drivers 11.8 or 12.x

# Run directly - no model conversion needed!
python simple_detector.py
```

**Performance**: 80-250 FPS | ~5-10ms latency | Works on all NVIDIA GPUs | No conversion required

### Intel GPU - OpenVINO Version

```bash
cd intel_gpu_detector

# Install dependencies
pip install -r requirements.txt

# Convert models to OpenVINO format (one-time setup)
python convert_model.py

# Run detection
python simple_detector.py
```

**Performance**: 55-60 FPS | ~17ms latency | Optimized for Intel integrated GPUs

### CPU Version (Testing)

```bash
cd old_version

# Install dependencies
pip install onnxruntime opencv-python numpy onnx

# Run video detection
python video_detector.py --video input.mp4
```

**Performance**: 20-40 FPS | ~35ms+ latency | For testing and video file analysis

## Use Case: Moonlight Game Streaming

This system is designed for detecting game objects when streaming VALORANT via Moonlight:

```
┌──────────────────┐              ┌──────────────────┐
│  PC1 (Game Host) │              │  PC2 (Detection) │
│                  │   Sunshine   │                  │
│  VALORANT Game   │─────────────►│  Moonlight       │
│                  │  Streaming   │  (Full-screen)   │
└──────────────────┘              │       ↓          │
                                  │  Detector        │
                                  │       ↓          │
                                  │  Output Distance │
                                  └──────────────────┘
```

**Important**: Moonlight must run in **full-screen mode** on PC2 for correct screen coordinate capture.

## Available Models

Models are located in the `v11moudle/` directory:

**ONNX Models** (for direct use or conversion):
- `val_kenny_ultra_256_v11s.onnx` - 256×256 (fastest, recommended for TensorRT)
- `val_kenny_ultra_320_v11s.onnx` - 320×320 (balanced)
- `kenny_ultra_416_v11s.onnx` - 416×416 (good accuracy)
- `kenny_ultra_640_v11s.onnx` - 640×640 (most accurate, recommended for ONNX GPU)

**PyTorch Models** (.pt files):
- `val_kenny_ultra_250619_v11n.pt` - Nano version (smallest)
- `val_kenny_ultra_250628_v11s.pt` - Small version
- Additional variants available

**Generated Models** (created locally):
- TensorRT engines (`.engine`) - Generated in `nvidia_gpu_detector/models/` (hardware-specific)
- OpenVINO IR (`.xml/.bin`) - Generated in `intel_gpu_detector/models/`
- ONNX GPU uses ONNX models directly without conversion

## Configuration

Each version can be configured by editing the `main()` function in `simple_detector.py`:

```python
# Example for TensorRT version
engine_path = "../models/val_kenny_ultra_256_v11s.engine"
center_size = 256        # Capture region size (pixels)
conf_threshold = 0.50    # Confidence threshold (0.0-1.0)
iou_threshold = 0.35     # NMS IoU threshold (0.0-1.0)
```

**Common Adjustments**:
- **Larger detection range**: Increase `center_size` (e.g., 320, 416, 640)
- **Higher precision**: Increase `conf_threshold` (e.g., 0.60) - fewer false positives
- **More detections**: Decrease `conf_threshold` (e.g., 0.40) - may increase false positives

## Output Format

The system outputs distance and offset information:

```
Distance: 106.4px | Offset: (+10.5, -40.2)
```

- **Distance**: Euclidean distance from detected head to screen center (crosshair)
- **Offset**: X and Y axis offset in pixels (positive = right/down, negative = left/up)

Detection object format:
```python
{
    'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
    'confidence': 0.85,         # Detection confidence (0.0-1.0)
    'class_id': 1,              # Class ID (0-4)
    'class_name': 'head'        # Human-readable class name
}
```

## Architecture Highlights

### TensorRT Version (nvidia_gpu_detector/)
- **~350 lines** of core code
- TRTYOLO library for simplified TensorRT integration
- FP16 precision optimization
- Hardware-specific compilation (not portable)
- Automatic postprocessing (NMS, coordinate conversion)

### ONNX GPU Version (onnx_gpu_detector/)
- **~500 lines** of core code
- ONNX Runtime with CUDA execution provider
- No model conversion required
- Vectorized NumPy postprocessing (60% faster)
- Cross-machine compatible (same CUDA version)

### Intel GPU Version (intel_gpu_detector/)
- **~400 lines** of core code
- OpenVINO with FP16 precision
- Vectorized NumPy postprocessing
- Zero-resize design (capture size = model input size)

### Old Version (old_version/)
- ONNX Runtime CPU execution
- Video file detection with visualization
- Letterbox preprocessing with padding
- Per-class NMS
- Useful for testing and development

## Performance Comparison

### Hardware Requirements

| Version | Minimum GPU | Recommended GPU | Additional Requirements |
|---------|-------------|-----------------|------------------------|
| TensorRT | RTX 2060 | RTX 3060+ | CUDA 11.8+, TensorRT 8.6+ |
| ONNX GPU | GTX 1050 | RTX 3060+ | CUDA 11.8+ or 12.x |
| OpenVINO | Intel UHD 630 | Intel Iris Xe | Intel GPU drivers |
| CPU | N/A | N/A | None |

### Speed Comparison

| GPU Model | TensorRT | ONNX GPU | OpenVINO | CPU |
|-----------|----------|----------|----------|-----|
| RTX 4070 | 350 FPS | 220 FPS | N/A | 30 FPS |
| RTX 3060 | 200 FPS | 150 FPS | N/A | 25 FPS |
| GTX 1650 | N/A | 80 FPS | N/A | 20 FPS |
| Intel i5 iGPU | N/A | N/A | 55 FPS | 25 FPS |

## Key Optimizations

1. **Center-Only Capture**: Only captures the screen region around the crosshair (256×256 to 640×640)
2. **Zero-Resize Design**: Capture size matches model input size (no preprocessing resize)
3. **Vectorized Postprocessing**: NumPy batch operations for 60% performance improvement
4. **GPU Acceleration**: FP16 precision for optimal performance
5. **Minimal Code**: ~350-500 lines per version for easy maintenance

## Video Detection (Old Version)

For analyzing recorded gameplay:

```bash
cd old_version

# Basic usage
python video_detector.py --video input.mp4

# Custom model and output
python video_detector.py --model ../v11moudle/kenny_ultra_416_v11s.onnx --video input.mp4 --output output.mp4

# Adjust thresholds
python video_detector.py --conf 0.3 --iou 0.5 --video input.mp4

# No display (faster processing)
python video_detector.py --video input.mp4 --no-display

# Display only (don't save)
python video_detector.py --video input.mp4 --no-save
```

**Controls**: Press `q` to quit, `p` to pause/resume

## Important Notes

- Model files (`.pt`, `.onnx`) are **not tracked in git** due to size (~5-40 MB each)
- TensorRT engines (`.engine`) are **hardware-specific** and must be generated on the target machine
- TensorRT engines are **not portable** between different GPUs or driver versions
- ONNX GPU version requires no conversion and is more portable
- Windows with DirectX is required for `dxcam` screen capture
- Ensure Moonlight runs in **full-screen mode** for correct coordinate mapping

## System Requirements

### All Versions
- **OS**: Windows 10/11 (for DirectX screen capture)
- **Python**: 3.8+
- **RAM**: 4GB minimum

### TensorRT Version
- **GPU**: NVIDIA RTX 2060 or better
- **VRAM**: 4GB minimum
- **Software**: CUDA 11.8+, cuDNN, TensorRT 8.6+

### ONNX GPU Version
- **GPU**: NVIDIA GTX 1050 or better
- **VRAM**: 2GB minimum
- **Software**: CUDA 11.8+ or 12.x drivers

### Intel GPU Version
- **GPU**: Intel UHD 630 or better
- **Software**: Latest Intel GPU drivers

## Troubleshooting

**Q: Why does TensorRT fail to convert the model?**
- Ensure CUDA and TensorRT are properly installed
- TensorRT must match your CUDA version
- Run conversion on the target machine where you'll use the engine

**Q: ONNX GPU is slow or using CPU**
- Check if `onnxruntime-gpu` is installed (not just `onnxruntime`)
- Verify CUDA drivers are installed: `nvidia-smi`
- Ensure CUDA version matches onnxruntime-gpu requirements

**Q: Intel GPU version shows "GPU not found"**
- Update Intel GPU drivers to the latest version
- Some older Intel CPUs don't support GPU acceleration
- Fallback to `device = "CPU"` in configuration

**Q: Detection doesn't work in Moonlight**
- Ensure Moonlight is running in **full-screen mode**
- Check that screen capture region is correctly calculated
- Verify model is loaded successfully (check console output)

## Project Structure

```
VALORANT/
├── nvidia_gpu_detector/      # TensorRT version (fastest)
│   ├── simple_detector.py
│   ├── yolo_detector_tensorrt.py
│   ├── convert_model.py
│   └── requirements.txt
├── onnx_gpu_detector/        # ONNX GPU version (easiest)
│   ├── simple_detector.py
│   ├── yolo_detector_onnx.py
│   └── requirements.txt
├── intel_gpu_detector/       # OpenVINO version (Intel)
│   ├── simple_detector.py
│   ├── yolo_detector_openvino.py
│   ├── convert_model.py
│   └── requirements.txt
├── old_version/              # CPU version (testing)
│   ├── video_detector.py
│   ├── yolo_detector.py
│   └── analyze_model.py
├── v11moudle/                # ONNX and PyTorch models
│   ├── *.onnx
│   └── *.pt
├── CLAUDE.md                 # Development guide
└── README.md                 # This file
```

## Contributing

This is a personal project for VALORANT object detection. Feel free to fork and adapt for your own use cases.

## License

This project is for educational and personal use only. VALORANT and related assets are property of Riot Games.

## Acknowledgments

- YOLOv11 model architecture
- TRTYOLO library for TensorRT integration
- OpenVINO toolkit for Intel GPU optimization
- dxcam for ultra-fast screen capture
- The computer vision and deep learning community

---

**Note**: This tool is designed for single-player practice and analysis only. Use responsibly and in accordance with Riot Games' terms of service.
