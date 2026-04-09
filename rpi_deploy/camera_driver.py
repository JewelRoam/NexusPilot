#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派小车摄像头驱动模块
Camera Driver Module for Raspberry Pi Robot Car

支持USB摄像头和Pi Camera
提供视频流、图像捕获和物体跟踪功能
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Callable, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
from queue import Queue, Empty

# 尝试导入OpenCV
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None
    print("[WARNING] OpenCV not available, camera functions disabled")

# Type-only imports (never executed at runtime)
if TYPE_CHECKING:
    import cv2
    import numpy as np

from .hardware_config import hardware_config


@dataclass
class FrameData:
    """帧数据容器"""
    frame: np.ndarray
    timestamp: float
    frame_number: int


class CameraDriver:
    """
    摄像头驱动类
    支持USB摄像头和视频流服务
    """
    
    def __init__(self, device_id: int = None):
        self.config = hardware_config.camera
        self.device_id = device_id if device_id is not None else self.config.device_id
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_queue: Queue = Queue(maxsize=2)  # 只保留最新2帧
        self._current_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        
        # 回调函数
        self._frame_callbacks: list = []
        
        if not CV2_AVAILABLE:
            print("[ERROR] Camera driver requires OpenCV")
    
    def open(self) -> bool:
        """
        打开摄像头
        
        Returns:
            是否成功打开
        """
        if not CV2_AVAILABLE:
            return False
        
        # 如果已经打开，先关闭
        if self._cap is not None:
            self.close()
        
        try:
            self._cap = cv2.VideoCapture(self.device_id)
            
            # 设置分辨率
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # 等待摄像头初始化
            time.sleep(0.5)
            
            if not self._cap.isOpened():
                print(f"[ERROR] Failed to open camera {self.device_id}")
                return False
            
            # 读取一帧测试
            ret, frame = self._cap.read()
            if not ret:
                print("[ERROR] Camera opened but cannot read frames")
                return False
            
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"[INFO] Camera opened: {actual_width}x{actual_height} @ {self.config.fps}fps")
            return True
            
        except Exception as e:
            print(f"[ERROR] Camera open failed: {e}")
            return False
    
    def close(self):
        """关闭摄像头"""
        self.stop_capture()
        if self._cap:
            self._cap.release()
            self._cap = None
            # 给系统时间释放设备
            time.sleep(0.5)
        print("[INFO] Camera closed")
    
    def start_capture(self):
        """启动后台捕获线程"""
        if self._is_running:
            return
        
        if not self._cap or not self._cap.isOpened():
            if not self.open():
                return
        
        self._is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print("[INFO] Camera capture started")
    
    def stop_capture(self):
        """停止捕获线程"""
        self._is_running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
    
    def _capture_loop(self):
        """后台捕获循环"""
        while self._is_running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                self._frame_count += 1
                self._current_frame = frame
                
                # 更新队列（保持最新）
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass
                
                try:
                    self._frame_queue.put_nowait(FrameData(
                        frame=frame.copy(),
                        timestamp=time.time(),
                        frame_number=self._frame_count
                    ))
                except:
                    pass
                
                # 执行回调
                for callback in self._frame_callbacks:
                    try:
                        callback(frame)
                    except Exception as e:
                        print(f"[WARNING] Frame callback error: {e}")
            else:
                time.sleep(0.001)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        获取最新帧
        
        Args:
            timeout: 等待超时时间
        
        Returns:
            图像帧或None
        """
        if not CV2_AVAILABLE:
            return None
        
        if not self._is_running:
            # 单帧模式
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                return frame if ret else None
            return None
        
        # 从队列获取
        try:
            frame_data = self._frame_queue.get(timeout=timeout)
            return frame_data.frame
        except Empty:
            return self._current_frame
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧（非阻塞）"""
        return self._current_frame.copy() if self._current_frame is not None else None
    
    def register_callback(self, callback: Callable[[np.ndarray], None]):
        """注册帧回调函数"""
        self._frame_callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[np.ndarray], None]):
        """注销帧回调函数"""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)
    
    def capture_image(self, filename: str = None) -> Optional[np.ndarray]:
        """
        捕获单张图片
        
        Args:
            filename: 保存路径，为None则不保存
        
        Returns:
            捕获的图像
        """
        frame = self.get_frame()
        if frame is not None and filename:
            cv2.imwrite(filename, frame)
            print(f"[INFO] Image saved: {filename}")
        return frame
    
    def get_resolution(self) -> Tuple[int, int]:
        """获取当前分辨率"""
        if self._cap:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (self.config.width, self.config.height)
    
    def set_resolution(self, width: int, height: int) -> bool:
        """设置分辨率（不关闭摄像头）"""
        if self._cap:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # 给摄像头时间应用设置
            time.sleep(0.3)
            return True
        return False
    
    def is_opened(self) -> bool:
        """检查摄像头是否已打开"""
        return self._cap is not None and self._cap.isOpened()
    
    def get_status(self) -> dict:
        """获取摄像头状态"""
        width, height = self.get_resolution()
        return {
            'opened': self.is_opened(),
            'capturing': self._is_running,
            'resolution': f"{width}x{height}",
            'fps_target': self.config.fps,
            'frame_count': self._frame_count,
            'device_id': self.device_id
        }
    
    def test_camera(self, duration: int = 5):
        """
        测试摄像头（显示预览窗口）
        
        Args:
            duration: 测试持续时间（秒）
        """
        if not CV2_AVAILABLE:
            print("[ERROR] OpenCV not available")
            return
        
        print(f"\n=== Camera Test ({duration}s) ===")
        print("Press 'q' to quit early")
        
        if not self.open():
            return
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                frame = self.get_frame()
                if frame is not None:
                    frame_count += 1
                    
                    # 添加信息文本
                    fps = frame_count / (time.time() - start_time)
                    info_text = f"FPS: {fps:.1f} | Frame: {frame_count}"
                    cv2.putText(frame, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Camera Test', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cv2.destroyAllWindows()
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Captured {frame_count} frames in {elapsed:.1f}s ({actual_fps:.1f} FPS)")
            print("=== Test Complete ===")


class VideoStreamServer:
    """
    MJPEG视频流服务器
    用于远程查看摄像头画面
    """
    
    def __init__(self, camera: CameraDriver, port: int = None):
        self.camera = camera
        self.port = port if port is not None else hardware_config.network.video_port
        self._server_thread: Optional[threading.Thread] = None
        self._is_running = False
    
    def start(self):
        """启动视频流服务器"""
        if self._is_running:
            return
        
        try:
            from flask import Flask, Response
            
            app = Flask(__name__)
            
            @app.route('/')
            def index():
                return '''
                <html>
                <head><title>RPi Car Camera</title></head>
                <body>
                    <h1>Raspberry Pi Car Camera Stream</h1>
                    <img src="/video_feed" width="640" height="480">
                </body>
                </html>
                '''
            
            @app.route('/video_feed')
            def video_feed():
                def generate():
                    while self._is_running:
                        frame = self.camera.get_current_frame()
                        if frame is not None:
                            _, jpeg = cv2.imencode('.jpg', frame)
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   jpeg.tobytes() + b'\r\n')
                        time.sleep(0.033)  # ~30fps
                
                return Response(generate(),
                              mimetype='multipart/x-mixed-replace; boundary=frame')
            
            self._is_running = True
            
            def run_server():
                app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False)
            
            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()
            
            print(f"[INFO] Video stream server started on http://0.0.0.0:{self.port}")
            
        except ImportError:
            print("[ERROR] Flask required for video streaming: pip install flask")
    
    def stop(self):
        """停止视频流服务器"""
        self._is_running = False
        print("[INFO] Video stream server stopped")


# 便捷函数
def create_camera_driver(device_id: int = 0) -> CameraDriver:
    """创建摄像头驱动实例"""
    return CameraDriver(device_id)


def create_video_stream(camera: CameraDriver, port: int = 8080) -> VideoStreamServer:
    """创建视频流服务器"""
    return VideoStreamServer(camera, port)


if __name__ == "__main__":
    # 独立测试
    print("Testing Camera Driver...")
    camera = create_camera_driver()
    
    try:
        camera.test_camera(duration=10)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        camera.close()
