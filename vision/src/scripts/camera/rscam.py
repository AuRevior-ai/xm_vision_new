# import cv2
# 这个文件是 RealSense 相机的封装，定义了类 RealSenseCamera 用于初始化、启动、捕获图像和停止相机等等操作

import pyrealsense2 as rs
import numpy as np

class RealSenseCamera():
    def __init__(self):
        self.pipeline = rs.pipeline()#pipeline的意思是数据流管道
        self.config = rs.config()#配置对象
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)#启用深度流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)#启用彩色流

    def start(self):
        self.pipeline.start(self.config)#启动数据流管道
        
    def capture(self) -> tuple[rs.video_frame, rs.depth_frame]:#捕获图像的方法，返回彩色帧和深度帧的元组
        frames = self.pipeline.wait_for_frames()#等待获取一组新的帧
        depth_frame = frames.get_depth_frame()#获取深度帧
        color_frame = frames.get_color_frame()#获取彩色帧
        return color_frame, depth_frame

    def get_color_frame(self) -> tuple[rs.video_frame, np.ndarray]:#获取彩色图像帧的方法，返回元组
        color_frame, _ = self.capture()#获取彩色帧，使用的是类内的capture方法
        color_image = np.asanyarray(color_frame.get_data())#将彩色帧转换为NumPy数组
        return color_frame, color_image#返回彩色帧对象和对应的NumPy数组表示的图像
    
    def get_depth_frame(self) -> tuple[rs.depth_frame, np.ndarray]:#获取深度图像帧的方法，返回元组
        _, depth_frame = self.capture()#获取深度帧，使用的是类内的capture方法
        depth_image = np.asanyarray(depth_frame.get_data())#将深度帧转换为NumPy数组,数组的大致形式为二维矩阵，每个元素表示对应像素点的深度值
        return depth_frame, depth_image#返回深度帧对象和对应的NumPy数组表示的图像

    def stop(self):
        self.pipeline.stop()
        
    def get_height_and_width(self):
        return 480, 640