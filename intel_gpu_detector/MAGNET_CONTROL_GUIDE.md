# 电磁铁控制功能使用指南

## 🎯 功能概述

基于迟滞比较器的智能电磁铁控制系统，为FPS游戏优化：
- ⚡ **零延迟设计**：无滤波、无缓冲，最快响应
- 🎮 **防抖动**：迟滞区间（25-35px）防止边界频繁切换
- 📡 **低网络负载**：仅在状态改变时发送UDP指令
- 🔌 **即插即用**：配置ESP32 IP即可启用

## 📋 系统要求

### PC端（检测系统）
- Python 3.8+
- 已安装依赖：`pip install -r requirements.txt`
- 与ESP32在同一局域网

### ESP32端（电磁铁控制器）
- ESP32开发板（任意型号）
- Arduino IDE 或 PlatformIO
- WiFi网络

## 🚀 快速开始

### 步骤1：配置ESP32

1. 打开 `ESP32_UDP_Receiver.ino`
2. 修改WiFi配置：
   ```cpp
   const char* ssid = "你的WiFi名称";
   const char* password = "你的WiFi密码";
   ```
3. （可选）修改电磁铁引脚：
   ```cpp
   const int MAGNET_PIN = 2;  // 默认GPIO2
   ```
4. 上传到ESP32
5. 打开串口监视器，记录ESP32的IP地址（如 `192.168.1.100`）

### 步骤2：配置PC端

编辑 `simple_detector.py` 的 `main()` 函数：

```python
CONFIG = {
    # ... 其他配置 ...

    # 启用电磁铁控制
    "esp32_ip": "192.168.1.100",  # 填入ESP32的IP地址
    "esp32_port": 8888,

    # 迟滞阈值配置
    "magnet_threshold_on": 25.0,   # 距离≤25px时开启
    "magnet_threshold_off": 35.0,  # 距离≥35px时关闭
}
```

### 步骤3：运行系统

```bash
cd intel_gpu_detector
python simple_detector.py
```

## ⚙️ 工作原理

### 迟滞比较器逻辑

```
                     关闭阈值 (35px)
                          |
    -------|--------------|--------------|-------
           |   保持区     |              |
           |   (迟滞)     |   关闭区     |
    -------|--------------|--------------|-------
           |              |
       开启区      开启阈值 (25px)
```

**状态转换规则**：
1. 距离 ≤ 25px → **开启电磁铁**
2. 距离 ≥ 35px → **关闭电磁铁**
3. 25px < 距离 < 35px → **保持当前状态**（防抖）
4. 无目标 → **立即关闭**（FPS优化）

### 通信流程

```
PC检测系统                ESP32控制器
    |                          |
    |  检测到头部，距离=20px    |
    |  -> 应开启电磁铁          |
    |                          |
    |--- UDP: 0x01 ----------->|
    |                          |  开启电磁铁
    |                          |
    |  距离=30px（迟滞区）      |
    |  -> 保持开启，不发送      |
    |                          |
    |  距离=40px                |
    |  -> 应关闭电磁铁          |
    |                          |
    |--- UDP: 0x02 ----------->|
    |                          |  关闭电磁铁
```

## 🔧 参数调优

### 调整阈值

根据你的游戏场景和硬件调整：

```python
# 更激进（更早触发，适合快速移动目标）
"magnet_threshold_on": 30.0,
"magnet_threshold_off": 40.0,

# 更保守（更精准触发，减少误触发）
"magnet_threshold_on": 20.0,
"magnet_threshold_off": 30.0,

# 更大迟滞区间（更强的防抖）
"magnet_threshold_on": 20.0,
"magnet_threshold_off": 45.0,
```

**建议**：
- 迟滞区间（差值）：`10-15px` 适合大多数场景
- 开启阈值：与你的屏幕`center_size`相关，建议 `center_size * 0.1`

### 响应模式

在 `magnet_controller.py` 中可配置：

```python
instant_off=True   # FPS推荐：无目标立即关闭
instant_off=False  # 延迟关闭，适合目标频繁遮挡的情况
```

## 📊 性能数据

| 操作 | 延迟 | 说明 |
|------|------|------|
| 检测到目标 | ~17ms | OpenVINO推理时间 |
| 状态判断 | <0.1ms | 迟滞比较器计算 |
| UDP发送 | <1ms | 局域网非阻塞发送 |
| ESP32处理 | <1ms | GPIO控制 |
| **总延迟** | **~20ms** | 从检测到电磁铁动作 |

**网络负载**：
- 典型场景：2-5次/秒（仅状态改变时）
- 最坏情况：60次/秒（如果每帧都切换，实际不会发生）
- 每次数据包：1字节

## 🛠️ 测试与调试

### 测试UDP通信

使用Python脚本测试：

```python
import socket

# 创建UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 发送测试指令
esp32_addr = ("192.168.1.100", 8888)

# 开启电磁铁
sock.sendto(b'\x01', esp32_addr)
print("发送：开启")

# 等待几秒
import time
time.sleep(2)

# 关闭电磁铁
sock.sendto(b'\x02', esp32_addr)
print("发送：关闭")

sock.close()
```

### 调试模式

使用带调试信息的控制器：

```python
from magnet_controller import MagnetControllerDebug

# 替换 MagnetController 为 MagnetControllerDebug
self.magnet = MagnetControllerDebug(...)
```

这将打印所有状态切换信息：
```
[Magnet] False -> True (开启)
[Magnet] True -> False (关闭)
```

## ⚠️ 注意事项

1. **网络要求**：
   - PC和ESP32必须在同一局域网
   - 建议使用5GHz WiFi减少延迟
   - 关闭防火墙的UDP 8888端口限制

2. **硬件安全**：
   - 确保电磁铁功率与ESP32匹配
   - 建议使用继电器或MOSFET驱动电磁铁
   - 不要直接连接大功率电磁铁到GPIO

3. **电源管理**：
   - 电磁铁可能需要独立供电
   - ESP32和电磁铁共地

4. **游戏性能**：
   - UDP发送使用非阻塞模式，不会影响检测FPS
   - 建议在实际游戏前测试性能影响

## 🔌 硬件连接示例

```
ESP32 (GPIO2) ----> 继电器/MOSFET ----> 电磁铁 (+)
                                              |
ESP32 (GND) <--------------------------------电磁铁 (-)
```

**推荐方案**：使用5V继电器模块
- ESP32 GPIO2 → 继电器信号
- ESP32 5V → 继电器VCC
- ESP32 GND → 继电器GND
- 电磁铁 → 继电器开关

## 📈 高级优化（可选）

如果需要更低延迟，可以考虑：

1. **使用ESP32的以太网模块**（有线连接，延迟<0.5ms）
2. **UDP组播**（如果有多个ESP32）
3. **状态预测**（基于目标移动轨迹）

但对于FPS游戏，当前方案的20ms总延迟已经足够快！

## 📝 故障排除

### 问题1：ESP32收不到指令

**检查**：
- ESP32的IP地址是否正确
- PC和ESP32是否在同一网络
- 防火墙是否阻止UDP 8888
- ESP32串口监视器是否显示"等待指令..."

**测试**：
```bash
# Windows: 测试网络连通性
ping 192.168.1.100
```

### 问题2：电磁铁不动作

**检查**：
- GPIO引脚接线是否正确
- 电磁铁是否需要独立供电
- 继电器是否正常工作
- ESP32串口是否显示状态改变

### 问题3：频繁抖动

**解决**：
- 增大迟滞区间：`threshold_off - threshold_on`
- 提高`conf_threshold`减少误检测
- 检查网络延迟是否过高

## 📧 支持

如有问题，请检查：
1. ESP32串口输出
2. PC端控制台输出
3. 网络连接状态

---

**祝游戏愉快！** 🎮
