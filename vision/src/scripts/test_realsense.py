#!/usr/bin/env python3
"""
RealSense 深度相机录制脚本
支持实时预览和保存深度帧、彩色帧
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import argparse
import json


class RealSenseRecorder:
    def __init__(self, width=640, height=480, fps=30):
        """
        初始化RealSense相机
        :param width: 图像宽度
        :param height: 图像高度
        :param fps: 帧率
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # 创建保存目录
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"realsense_data_{self.timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "color"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "depth_raw"), exist_ok=True)
        
        # 配置相机管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置彩色和深度流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # 对齐对象，用于对齐深度和彩色图像
        self.align = rs.align(rs.stream.color)
        
        self.frame_count = 0
        self.is_recording = False
        
    def start_camera(self):
        """启动相机"""
        try:
            # 启动管道
            profile = self.pipeline.start(self.config)
            
            # 获取深度传感器并设置一些参数
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print(f"深度比例: {depth_scale}")
            
            # 保存相机内参
            self.save_camera_info(profile)
            
            return True
        except Exception as e:
            print(f"启动相机失败: {e}")
            return False
    
    def save_camera_info(self, profile):
        """保存相机内参信息"""
        # 获取内参
        color_stream = profile.get_stream(rs.stream.color)
        depth_stream = profile.get_stream(rs.stream.depth)
        
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        # 获取外参
        depth_to_color_extrin = depth_stream.get_extrinsics_to(color_stream)
        
        camera_info = {
            "color_intrinsics": {
                "width": color_intrinsics.width,
                "height": color_intrinsics.height,
                "fx": color_intrinsics.fx,
                "fy": color_intrinsics.fy,
                "ppx": color_intrinsics.ppx,
                "ppy": color_intrinsics.ppy,
                "coeffs": color_intrinsics.coeffs
            },
            "depth_intrinsics": {
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "ppx": depth_intrinsics.ppx,
                "ppy": depth_intrinsics.ppy,
                "coeffs": depth_intrinsics.coeffs
            },
            "depth_to_color_extrinsics": {
                "rotation": depth_to_color_extrin.rotation,
                "translation": depth_to_color_extrin.translation
            }
        }
        
        with open(os.path.join(self.save_dir, "camera_info.json"), 'w') as f:
            json.dump(camera_info, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    def run(self):
        """主运行循环"""
        if not self.start_camera():
            return
        
        print("RealSense 深度相机已启动")
        print("按键说明:")
        print("  空格键 - 开始/停止录制")
        print("  's' - 保存当前帧")
        print("  'q' - 退出程序")
        print(f"数据保存路径: {self.save_dir}")
        
        try:
            while True:
                # 等待一组连贯的帧
                frames = self.pipeline.wait_for_frames()
                
                # 对齐深度帧到彩色帧
                aligned_frames = self.align.process(frames)
                
                # 获取对齐的帧
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # 转换为numpy数组
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # 创建深度的彩色图像用于显示
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # 水平拼接图像用于显示
                images = np.hstack((color_image, depth_colormap))
                
                # 添加文本信息
                text = f"Frame: {self.frame_count}"
                if self.is_recording:
                    text += " [RECORDING]"
                cv2.putText(images, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow('RealSense Color & Depth', images)
                
                # 如果正在录制，保存帧
                if self.is_recording:
                    self.save_frame(color_image, depth_image)
                    self.frame_count += 1
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # 空格键切换录制状态
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        print("开始录制...")
                        self.frame_count = 0
                    else:
                        print("停止录制")
                elif key == ord('s'):  # 保存当前帧
                    self.save_frame(color_image, depth_image)
                    print(f"已保存当前帧")
                
        finally:
            # 清理资源
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print(f"录制完成，共保存 {self.frame_count} 帧")
    
    def save_frame(self, color_image, depth_image):
        """保存单帧图像"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # 保存彩色图像
        color_filename = os.path.join(self.save_dir, "color", f"color_{timestamp}.png")
        cv2.imwrite(color_filename, color_image)
        
        # 保存深度图像（可视化版本）
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_filename = os.path.join(self.save_dir, "depth", f"depth_{timestamp}.png")
        cv2.imwrite(depth_filename, depth_colormap)
        
        # 保存原始深度数据
        depth_raw_filename = os.path.join(self.save_dir, "depth_raw", f"depth_raw_{timestamp}.npy")
        np.save(depth_raw_filename, depth_image)


def main():
    parser = argparse.ArgumentParser(description='RealSense 深度相机录制工具')
    parser.add_argument('--width', type=int, default=640, help='图像宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480, help='图像高度 (默认: 480)')
    parser.add_argument('--fps', type=int, default=30, help='帧率 (默认: 30)')
    
    args = parser.parse_args()
    
    recorder = RealSenseRecorder(args.width, args.height, args.fps)
    recorder.run()


if __name__ == "__main__":
    main()
