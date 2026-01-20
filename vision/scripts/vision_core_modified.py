#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Modified version for running without specialized hardware
Author: Modified for compatibility
Date: 2025-8-20
Description: 兼容版本的视觉节点，支持普通摄像头运行
'''

import os
import sys
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import *

# 尝试导入专用摄像头库，如果失败则使用普通摄像头
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    print("RealSense not available, using OpenCV camera")
    REALSENSE_AVAILABLE = False

try:
    from pyk4a import ColorResolution, Config, PyK4A
    KINECT_AVAILABLE = True
except ImportError:
    print("Azure Kinect not available, using OpenCV camera")
    KINECT_AVAILABLE = False

from xm_vision.srv import (VisionToSmach, VisionToSmachRequest,
                           VisionToSmachResponse)

# 兼容版本的摄像头类
class camera_opencv():
    def __init__(self):
        # 使用普通摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Warning: Cannot open camera")
            
    def get_param(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Cannot read from camera")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        width = 640   # 设置图像宽度
        height = 480  # 设置图像高度
        TorF = True
        paramlist = {}   

        paramlist[0] = 1.0  # fx 缩放因子
        paramlist[1] = 1.0  # fy 缩放因子
        paramlist[2] = frame
        # 模拟深度点云数据
        paramlist[3] = np.zeros((height, width, 3), dtype=np.float32)
        paramlist[4] = TorF
        return paramlist

# 原始摄像头类（如果硬件可用）
class camera():
    def __init__(self):
        if KINECT_AVAILABLE:
            config = Config(color_resolution=ColorResolution.RES_1080P)
            self.camera = PyK4A(config)
            self.camera.start()
        else:
            # 降级到普通摄像头
            self.opencv_cam = camera_opencv()

    def get_param(self):
        if KINECT_AVAILABLE:
            capture = self.camera.get_capture()
            width = 2600   
            height = 1600  
            TorF = True
            paramlist = {}   
            paramlist[0] = width / capture.color.shape[1]
            paramlist[1] = height / capture.color.shape[0]
            paramlist[2] = capture.color
            paramlist[3] = capture.transformed_depth_point_cloud
            paramlist[4] = TorF
            return paramlist
        else:
            return self.opencv_cam.get_param()

class camera_small():
    def __init__(self):
        if REALSENSE_AVAILABLE:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
            self.pipeline.start(config)
        else:
            # 降级到普通摄像头
            self.opencv_cam = camera_opencv()

    def get_param(self):
        if REALSENSE_AVAILABLE:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            paramlist = {}
            paramlist[0] = 1.0
            paramlist[1] = 1.0  
            paramlist[2] = color_image
            paramlist[3] = np.zeros((480, 640, 3), dtype=np.float32)
            paramlist[4] = True
            return paramlist
        else:
            return self.opencv_cam.get_param()

# 视觉功能模块的导入（需要根据实际情况注释掉不可用的模块）
try:
    sys.path.append('/home/xiaomeng/catkin_ws/src/xm_vision/src/scripts')
    from normal import FollowPeople
    from takephoto import RememberFace
    VISION_MODULES_AVAILABLE = True
except ImportError:
    print("Some vision modules not available")
    VISION_MODULES_AVAILABLE = False

# 简化版的响应函数
def call_back_1(req):
    xm_res = VisionToSmachResponse()
    
    if req.command == 1:  # 跟随功能
        if VISION_MODULES_AVAILABLE:
            try:
                paramlist = xm_camera.get_param()
                frame, coordinate = xm_follow.use(paramlist[0], paramlist[1], paramlist[2], paramlist[3], paramlist[4])
                xm_res.info1 = coordinate[0]
                xm_res.info2 = coordinate[1]
                xm_res.info3 = coordinate[2]
            except Exception as e:
                print(f"Error in follow function: {e}")
                xm_res.info1 = 0.0
                xm_res.info2 = 0.0
                xm_res.info3 = 0.0
        else:
            # 返回模拟数据
            xm_res.info1 = 1.0
            xm_res.info2 = 0.0
            xm_res.info3 = 2.0
            xm_res.info4 = "follow_simulation"
    
    # 其他功能类似处理...
    else:
        xm_res.info4 = "function_not_available"
    
    print(f"Response: {xm_res}")
    return xm_res

if __name__=="__main__":
    # 初始化摄像头
    print("Initializing cameras...")
    xm_camera = camera()
    xm_camera_small = camera_small()
    
    # 初始化视觉模块
    if VISION_MODULES_AVAILABLE:
        try:
            xm_follow = FollowPeople()
            print("Vision modules loaded successfully")
        except Exception as e:
            print(f"Error loading vision modules: {e}")
    
    # 初始化ROS节点
    rospy.init_node("vision_core")
    s = rospy.Service("vision_to_smach", VisionToSmach, call_back_1)
    rospy.loginfo('Vision core started (compatibility mode)')
    rospy.spin()
