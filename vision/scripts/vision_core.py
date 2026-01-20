#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: Xiaomeng 
Date: 2025-1-16
LastEditTime: 2025-4-8
LastEditors: Xiaomeng 
Description: 现役视觉节点,核心的视觉节点,和其余部分进行通信的接口
'''

import os

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import *
from pyk4a import ColorResolution, Config, PyK4A#
from xm_vision.srv import (VisionToSmach, VisionToSmachRequest,
                           VisionToSmachResponse)

sys.path.append('/home/xiaomeng/catkin_ws/src/xm_vision/src/scripts')
# 从 WhatIsThat.GestureRecognizer 导入 GestureRecognizer
# 从 WhatIsThat.yolodetect 导入 yolodetect
from bag import PointLuggage
from emptychair import EmptyChair
from face_identification.face_identification import FaceIdentifier
from find_infraction.drink import FindPeopleDrink
from find_infraction.people import FindPeople
from find_infraction.shoes import FindPeopleNoShoes
from find_object.desk_serve_breakfast import DeskObject
from find_object.desk_storing_groceries import DeskObject2
from find_object.desk_clean_the_table import DeskObject_clean
from find_object.shelf_origin import ShelfObjiectDetection
from find_object.shelf_nohave import ShelfObjiectDetection2
from find_object.shelf import ShelfObjectDetection
from posture_identification.posture_identification import PostureIdentifier
from normal import FollowPeople
from takephoto import RememberFace


sys.path.append('/home/xiaomeng/catkin_ws/src/xm_smach/xm_smach/smach_lib')
from new_smach_special.various_dic import *


# 摄像头类，用于开启摄像头
# get_param从摄像头得到参数并返回，以满足后续视觉函数的需要
class camera():
    def __init__(self):
        # 开启摄像头
        config = Config(color_resolution=ColorResolution.RES_1080P)#配置
        self.camera = PyK4A(config)#配置azure kinect相机
        self.camera.start()#开启相机

    def get_param(self):
        capture = self.camera.get_capture()#获取图像

        width = 2600   # 设置图像宽度
        height = 1600  # 设置图像高度
        TorF = True#判断是否成功获取图像的标志位
        paramlist = {}   #存储参数的字典

        paramlist[0] = width / capture.color.shape[1]#横向缩放因子
        paramlist[1] = height / capture.color.shape[0]#纵向缩放因子
        paramlist[2] = capture.color#彩色图像,这里是一个numpy数组
        paramlist[3] = capture.transformed_depth_point_cloud#深度点云,这里是一个numpy数组
        print(paramlist[3].shape)#打印深度点云的形状,一般来说结果是(307200, 3),意思是有307200个点,每个点有x,y,z三个坐标
        paramlist[4] = TorF#是否成功获取图像的标志位
        return paramlist#返回参数字典

# 小摄像头类，用于开启小摄像头
# get_param从摄像头得到参数并返回，以满足后续视觉函数的需要
class camera_small():
    def __init__(self):
        self.pipeline = rs.pipeline()#开启管道,rs的意思是realsense深度相机
        config = rs.config()#配置
        config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)#允许深度流
        config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)#允许彩色流
        self.pipeline.start(config)#开启管道

    def get_param(self):#从小摄像头获取参数
        paramlist = {}#存储参数的字典
        frames = self.pipeline.wait_for_frames()#等待获取一帧数据
        depth_frame = frames.get_depth_frame()#获取深度帧
        color_frame = frames.get_color_frame()#获取彩色帧
        paramlist[0] = np.asanyarray(color_frame.get_data())#彩色帧
        #paramlist[0] = cv2.resize(paramlist[0],(1280, 720))
        depth_image = np.asanyarray(depth_frame.get_data())#深度帧
        #depth_image = cv2.resize(depth_image,(1280, 720))
        #生成点云
        point_cloud = rs.pointcloud()#获取点云对象
        points = point_cloud.calculate(depth_frame)#计算点云
        vertices = np.asanyarray(points.get_vertices())#获取点云的顶点
        paramlist[1] = vertices.view(np.float32).reshape(vertices.shape + (-1,))#深度点云
        #paramlist[1] = np.repeat(paramlist[1],3,axis=0)
        width = 2600  # 设置图像宽度
        height = 1600  # 设置图像高度
        paramlist[2] = width / paramlist[0].shape[1]#缩放因子
        paramlist[3] = height / paramlist[0].shape[0]#缩放因子
        return paramlist

# 两个摄像头的实例化
xm_camera = camera()
xm_camera_small = camera_small()

# 视觉功能的实例化，处理不同的任务时，可以将不需要的注释掉
xm_follow = FollowPeople()
# xm_gesture = GestureRecognizer()
# xm_detect = yolodetect()

#三个yolo模型路径
yolo_1 = "/home/xiaomeng/catkin_ws/src/xm_vision/src/scripts/find_object/best.pt"
yolo_2 = '/home/xiaomeng/桌面/visionpart/yolov8n.pt'
yolo_3 = '/home/xiaomeng/catkin_ws/src/xm_vision/src/scripts/find_infraction/best.pt'


xm_luggage = PointLuggage(yolo_1)
xm_seat = EmptyChair(yolo_2)
xm_face_identify = FaceIdentifier()

xm_find_people_no_shoes = FindPeopleNoShoes(yolo_2,yolo_3)
xm_find_people_drink = FindPeopleDrink(yolo_2)
xm_find_people = FindPeople(yolo_2)
xm_pos = PostureIdentifier()

list = [481, 481, 481, 481, 481, 481, 481, 481, 481, 481]

def call_back_1(req):#回调函数,处理请求,smach的意思是状态机
    global list#定义全局变量list
    xm_res = VisionToSmachResponse()#定义路径srcc\xm_vision\srv\VisionToSmach.srv
    # 根据不同的指令，调用不同的视觉功能
    
    if(req.command==1):# 跟随功能，此时返回摄像头识别出的人的三个坐标，详问视觉学姐
        paramlist = xm_camera.get_param()#获取参数字典
        frame, coordinate = xm_follow.use(paramlist[0], paramlist[1], paramlist[2], paramlist[3], paramlist[4])
        """
        四个参数的意思分别是:
        paramlist[0]: 彩色图像
        paramlist[1]: 深度点云
        paramlist[2]: 横向缩放因子 
        paramlist[3]: 纵向缩放因子
        paramlist[4]: 是否成功获取图像的标志位
    
        use函数的返回值:
        frame: 处理后的图像
        coordinate: 识别出的人在摄像头坐标系下的三个坐标
        """
        xm_res.info1 = coordinate[0]
        xm_res.info2 = coordinate[1]
        xm_res.info3 = coordinate[2]
    # # 手势识别功能，返回手势名字字符串    
    # elif(req.command==2):
    #     frame = xm_camera.get_param()
    #     outcome = xm_gesture.use(frame)
    #     xm_res.info4 = outcome
    # # 物品识别功能，返回物品名字字符串
    # elif(req.command==3):
    #     frame = xm_camera_small.get_param()
    #     outcome = xm_detect.use(frame) 
    #     xm_res.info4 = outcome
    # 手指袋子识别功能(carry_my_luggage), 返回所指的袋子的三个坐标
    elif(req.command==4): 
        paramlist = xm_camera_small.get_param()  
        outcome1,outcome2,outcome3,_ = xm_luggage.use(paramlist[0], paramlist[1], paramlist[2], paramlist[3])
        xm_res.info1 = outcome1
        xm_res.info2 = outcome2
        xm_res.info3 = outcome3

    # 人脸录入功能(Receptionist) ，录入人脸并自动训练模型，结束后函数返回1
    # input1的字符串标记着我此时录入的脸对应的人的编号
    elif(req.command==5): 
        xm_remem_face = RememberFace()
        # xm_remem_face.start_camera()
        while True:
            paramlist = xm_camera.get_param()
            outcome = xm_remem_face.capture_faces(paramlist[2],req.input1)
            print(outcome)
            if outcome == 'succeed':
                break
        xm_res.info4 = outcome
        os.system("python3 /home/xiaomeng/catkin_ws/src/xm_vision/src/scripts/face_identification/preprocess.py")
        
    # 识别空座位功能(Receptionist), 返回    
    elif(req.command==6):
        paramlist = xm_camera.get_param()    
        outcome1, _ = xm_seat.use(paramlist[2], paramlist[3], paramlist[0], paramlist[1])
        xm_res.info1 = outcome1            

    # 识别人功能(Receptionist)            
    elif(req.command==7):
        if(req.input1 == 'over'):
            # del xm_face_identify
            pass
        else:
            paramlist = xm_camera.get_param()  
            outcome1,outcome2,_= xm_face_identify.use(paramlist[2],req.input1)
            xm_res.info1 = outcome1
            xm_res.info2 = outcome2
    # 识别是否有人穿鞋了
    elif(req.command==8):
        paramlist = xm_camera.get_param()    
        outcome1,outcome2,outcome3,outcome4,_ = xm_find_people_no_shoes.use(paramlist[2],paramlist[3],paramlist[0],paramlist[1])
        xm_res.info4 = outcome1
        xm_res.info1 = outcome2
        xm_res.info2 = outcome3
        xm_res.info3 = outcome4

    # 识别是否有人没拿饮料
    elif(req.command==9):
        paramlist = xm_camera.get_param()    
        outcome1,outcome2,outcome3,outcome4,_ = xm_find_people_drink.use(paramlist[2],paramlist[3],paramlist[0],paramlist[1])
        xm_res.info4 = outcome1
        xm_res.info1 = outcome2
        xm_res.info2 = outcome3
        xm_res.info3 = outcome4
    
    # 识别是否存在人
    elif(req.command==10):
        paramlist = xm_camera.get_param()    
        outcome1,outcome2,outcome3,outcome4,_ = xm_find_people.use(paramlist[2],paramlist[3],paramlist[0],paramlist[1])
        xm_res.info4 = outcome1
        xm_res.info1 = outcome2
        xm_res.info2 = outcome3
        xm_res.info3 = outcome4

    # 识别自己训练的物品， 返回物品字符串与坐标（相对于摄像头）
    elif(req.command==11):
        # count = 0
        # count2 = 0
        # flag_1 = 0
        # while True:
        #     if(count == 10)or(count2 == 10):
        #         break
        #     paramlist = xm_camera_small.get_param()  
        #     outcome1,outcome2,outcome3,outcome4, y = DeskObject(object_dic[req.input1], paramlist[0], paramlist[1], paramlist[2], paramlist[3], 0 ,0)
        #     if(y != 481):
        #         flag_1 = 0
        #         list[count] = y 
        #         count += 1
        #     else:
        #         if flag_1 == 1:
        #             count2 += 1
        #         else:
        #             count2 = 0
        #         flag_1 = 1
        # real_y = sum(list) / 10
        paramlist = xm_camera_small.get_param()  
        outcome1,outcome2,outcome3,outcome4, y = DeskObject_clean(object_dic[req.input1], paramlist[0], paramlist[1], paramlist[2], paramlist[3] ,1)
        xm_res.info4 = object_dic_reverse[outcome1]
        xm_res.info1 = outcome4
        xm_res.info2 = -outcome2
        xm_res.info3 = -outcome3
        list = [481, 481, 481, 481, 481, 481, 481, 481, 481, 481]

    # 识别柜子上的物品, 返回柜子层数
    elif(req.command==12):
        paramlist = xm_camera.get_param()  
        outcome1,_ = ShelfObjiectDetection(classify_dic[req.input1], paramlist[2], paramlist[3], paramlist[0], paramlist[1])
        xm_res.info1 = outcome1
    
    elif(req.command==13):
        paramlist = xm_camera.get_param()  
        outcome1 = ShelfObjiectDetection2(paramlist[2])
        xm_res.info4 = outcome1
    print(xm_res)
    return xm_res
    

if __name__=="__main__":
    rospy.init_node("vision_core")
    s = rospy.Service("vision_to_smach", VisionToSmach, call_back_1)
    rospy.loginfo('ouyear')
    rospy.spin()

