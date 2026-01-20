#此脚本用于检测行李并返回其坐标，坐标格式为（z，x，y），单位为米，三维坐标
from bag import PointLuggage
import cv2
from pyk4a import PyK4A, ColorResolution, Config
yolo_model_path = "D:/pycharm/codepython/ultralytics-main/weights/yolov8n.pt"
estimater = PointLuggage(yolo_model_path)#加载行李箱检测模型，其中函数返回三维坐标
config = Config(color_resolution=ColorResolution.RES_1080P)#设置相机分辨率
camera = PyK4A(config)#初始化相机
camera.start()
while True:
    capture = camera.get_capture()
    width = 2600  # 设置图像宽度
    height = 1600  # 设置图像高度
    fx = width / capture.color.shape[1]
    fy = height / capture.color.shape[0]
    color_image = capture.color
    transformed_depth_point_cloud = capture.transformed_depth_point_cloud
    crop_frme,frame, x, y, z = estimater.use(color_image, transformed_depth_point_cloud, fx, fy)
    cv2.imshow('crop_frame', crop_frme)
    cv2.imshow('frame', frame)
    print(x, y, z)
    if cv2.waitKey(1) in [ord('q'), ord('Q')]:
        exit(0)
'''
from emptychair import EmptyChair
import cv2
from pyk4a import PyK4A, ColorResolution, Config
yolo_model_path = "D:/pycharm/codepython/ultralytics-main/weights/yolov8n.pt"
estimater = EmptyChair(yolo_model_path)
config = Config(color_resolution=ColorResolution.RES_1080P)
camera = PyK4A(config)
camera.start()
while True:
    capture = camera.get_capture()
    width = 2600  # 设置图像宽度
    height = 1600  # 设置图像高度
    fx = width / capture.color.shape[1]
    fy = height / capture.color.shape[0]
    color_image = capture.color
    transformed_depth_point_cloud = capture.transformed_depth_point_cloud
    crop_image,color_image,x,y,z = estimater.use(color_image, transformed_depth_point_cloud, fx, fy)
    cv2.imshow('crop_image', crop_image)
    cv2.imshow('frame', color_image)
    print(x, y, z)
    if cv2.waitKey(1) in [ord('q'), ord('Q')]:
        exit(0)
'''