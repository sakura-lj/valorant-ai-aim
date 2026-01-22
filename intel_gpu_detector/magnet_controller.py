"""
电磁铁控制器 - 基于迟滞比较器，为FPS游戏优化
零延迟设计，快速响应
"""

import socket
import time
from typing import Optional


class MagnetController:
    """
    迟滞比较器电磁铁控制器

    设计目标：
    - 零额外延迟（无滤波、无缓冲）
    - 快速响应（适合FPS游戏）
    - 防止边界抖动（迟滞区间）
    - 最小化网络负载（状态跟踪）
    """

    # 指令定义
    CMD_ON = b'\x01'   # 开启电磁铁
    CMD_OFF = b'\x02'  # 关闭电磁铁

    def __init__(self,
                 esp32_ip: str,
                 esp32_port: int = 3333,
                 threshold_on: float = 25.0,   # 开启阈值（像素）
                 threshold_off: float = 35.0,  # 关闭阈值（像素）
                 instant_off: bool = True):    # 无目标时立即关闭
        """
        初始化电磁铁控制器

        Args:
            esp32_ip: ESP32的IP地址
            esp32_port: ESP32的UDP端口
            threshold_on: 开启阈值，距离 ≤ 此值时开启电磁铁
            threshold_off: 关闭阈值，距离 ≥ 此值时关闭电磁铁
            instant_off: True=无目标立即关闭（FPS推荐），False=延迟关闭
        """
        if threshold_on >= threshold_off:
            raise ValueError("开启阈值必须小于关闭阈值（迟滞区间）")

        self.esp32_addr = (esp32_ip, esp32_port)
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off
        self.instant_off = instant_off

        # 状态跟踪
        self.magnet_state = False  # False=关闭, True=开启

        # UDP Socket（非阻塞）
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)  # 非阻塞模式，防止卡顿

        print(f"[MagnetController] 初始化完成")
        print(f"  - ESP32地址: {esp32_ip}:{esp32_port}")
        print(f"  - 开启阈值: {threshold_on}px")
        print(f"  - 关闭阈值: {threshold_off}px")
        print(f"  - 迟滞区间: {threshold_off - threshold_on}px")
        print(f"  - 响应模式: {'即时关闭' if instant_off else '延迟关闭'}")

    def update(self, distance: Optional[float]) -> bool:
        """
        更新控制状态（每帧调用）

        Args:
            distance: 目标距离（像素），None表示无目标

        Returns:
            当前电磁铁状态 (True=开启, False=关闭)
        """
        # 1. 无目标处理
        if distance is None:
            if self.instant_off and self.magnet_state:
                self._set_state(False)
            return self.magnet_state

        # 2. 迟滞比较器逻辑
        if distance <= self.threshold_on:
            # 距离足够近，应该开启
            if not self.magnet_state:
                self._set_state(True)
                print(f"[Magnet] 目标接近 ({distance:.1f}px)，开启电磁铁")

        elif distance >= self.threshold_off:
            # 距离足够远，应该关闭
            if self.magnet_state:
                self._set_state(False)
                print(f"[Magnet] 目标远离 ({distance:.1f}px)，关闭电磁铁")

        # 在迟滞区间内（threshold_on < distance < threshold_off）：保持原状态

        return self.magnet_state

    def _set_state(self, new_state: bool):
        """内部方法：设置电磁铁状态并发送UDP指令"""
        if new_state == self.magnet_state:
            return  # 状态未改变，不发送

        self.magnet_state = new_state
        cmd = self.CMD_ON if new_state else self.CMD_OFF

        try:
            # 非阻塞发送，不会卡顿主循环
            self.sock.sendto(cmd, self.esp32_addr)
        except BlockingIOError:
            # 缓冲区满，跳过本次发送（极少发生）
            pass
        except Exception as e:
            print(f"[Magnet] UDP发送失败: {e}")

    def force_off(self):
        """强制关闭电磁铁（清理时调用）"""
        if self.magnet_state:
            self.magnet_state = True  # 临时设置为True，确保_set_state会发送
            self._set_state(False)

    def close(self):
        """关闭控制器"""
        self.force_off()
        self.sock.close()

    def get_state(self) -> bool:
        """获取当前电磁铁状态"""
        return self.magnet_state


# 可选：带调试信息的版本
class MagnetControllerDebug(MagnetController):
    """
    调试版本：打印状态切换信息
    建议：正式使用时用基础版本，减少print开销
    """

    def _set_state(self, new_state: bool):
        """重写：增加调试输出"""
        if new_state == self.magnet_state:
            return

        old_state = self.magnet_state
        super()._set_state(new_state)

        state_str = "开启" if new_state else "关闭"
        print(f"[Magnet] {old_state} -> {new_state} ({state_str})")
