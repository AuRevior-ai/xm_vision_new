# vision
机器人“晓萌”的视觉模块

## 项目概览
- 提供人脸识别、朝向估计、手势识别、指向识别、姿态识别与通用目标检测等能力，支撑机器人视觉感知。
- 各子模块统一提供 `use(frame, ...)` 接口，方便封装为 ROS 节点。
- 摄像头抽象兼容 Azure Kinect、Intel RealSense 以及普通 USB 摄像头，适配多种硬件。

## 目录结构
```
D:.
├─.vscode                      # VS Code 工作区配置
└─vision
    ├─.git                     # Git 元数据及子模块镜像
    ├─README.md                # 本说明文档
    ├─realsease.py             # RealSense 测试脚本
    ├─split_dataset.py         # YOLO 数据集划分工具
    ├─split_dataset.py         # 原始目录结构中存在的重复条目
    ├─split_yolo_dataset.py    # 其他数据集辅助脚本
    └─src
        └─scripts
            ├─camera           ️ # 摄像头抽象模块
            │   ├─k4acam.py     # Azure Kinect 摄像头封装
            │   ├─opencvcam.py  # OpenCV 摄像头封装
            │   ├─rscam.py      # RealSense 摄像头封装
            │   └─__init__.py   # 摄像头工厂方法，进行初始化
            ├─face_identification   # 人脸识别模块
            │   ├─assets        # OpenFace 嵌入模型等资源,其中文件openface_nn4.small2.v1.t7为OpenFace模型文件
            │   ├─dataset       # 人脸数据集,主要数据集是琳可学姐自己(滑稽)
            │   ├─face_detection_model # 人脸检测模型文件,其中deploy.prototxt.txt为模型结构成文件,res10_300x300_ssd_iter_140000.caffemodel为模型权重文件
            │   ├─output        # 训练好的人脸嵌入与分类器,是一些pickle文件,也就是
            │   ├─scripts       # 离线特征提取与模型训练脚本,以script为名字的问及那加一般存储可运行的功能脚本
            │   └─face_identification.py   # 人脸识别主逻辑入口文件
            │   └──preprocess.py  # 人脸预处理脚本,运行脚本extract_embeddings.py和train_model.py,获得模型文件
            ├─face_orientation_recognition # 人脸朝向识别模块
            │   ├─modules       # 包含文件face_detection.py、mark_detection.py、pose_estimation.py等子模块,用于实现人脸检测、关键点检测与姿态估计功能
            │   ├─assets        # ONNX 模型文件,这是一种表示深度学习模型的开放格式  
            │   └─face_orientation.py  # 人脸朝向识别主逻辑入口文件,用于识别人脸相对于摄像头的朝向
            ├─gesture_recognition  # 手势识别模块
            │   ├─models        # 手势识别模型文件,其中gesture_recognizer.task为手势识别模型文件,task文件是MediaPipe Tasks的一种模型文件格式
            │   ├─utils         # 可视化工具脚本,用于绘制手部关键点与批量可视化
            │   └─gesture_recognition.py  # 手势识别主逻辑入口文件
            ├─pointing_gesture_recognition    # 指向识别模块
            │   ├─model    # 指向动作分类器模型文件,其中pointing_gesture_classifier.tflite为基于TensorFlow Lite的指向动作分类器模型
            │   ├─utils
            │   └─pointing_recognition.py   # 指向识别主逻辑入口文件，定义了PointingRecognizer类
            ├─posture_identification   # 姿态识别模块
            │   ├─dataset / models / output / scripts
            │   └─posture_identification.py   # 姿态识别主逻辑入口文件，定义了PostureIdentifier类
            ├─yolo_detection
            │   ├─utils
            │   └─yolo_detection.py   # YOLO 目标检测主逻辑入口文件，定义了YoloDetector类
            └─[test_*.py, camera_opencv*.py, takephoto.py 等]  # 演示与测试脚本
```
> 说明：`.git/modules/src/scripts/...` 目录为各视觉子模块的 Git 子模块镜像，源码实际位于 `src/scripts/`。

## 模块解析

### 摄像头抽象（`src/scripts/camera`）
- `__init__.py` 提供 `create_camera(camera_type)` 工厂，支持 `"realsense"`、`"azure_kinect"` 以及数值设备索引。
- `rscam.py` 基于 `pyrealsense2` 返回 RealSense 彩色与深度帧，附带分辨率信息。
- `k4acam.py` 通过 `pyk4a` 获取 Azure Kinect 的同步彩色与深度数据，默认使用广视场深度模式。
- `opencvcam.py` 提供 OpenCV 摄像头简易封装，目前 `capture()` 为待扩展占位实现。

### 人脸识别（`src/scripts/face_identification`）
- `face_identification.py` 定义 `FaceIdentifier`，加载 SSD 检测器、OpenFace 嵌入模型与 SVM 分类器（`output/recognizer`、`output/le.pickle`），输出识别结果及人脸中心的归一化坐标。
- `assets/` 存放 OpenFace `.t7` 嵌入模型；`face_detection_model/` 为 Caffe 检测模型；`dataset/` 为按人员划分的训练图片；`output/` 保存嵌入与分类器；`scripts/` 提供离线特征提取与模型训练脚本。
- `save_personal_faces.py` 支持快速新增人脸样本。

### 人脸朝向识别（`src/scripts/face_orientation_recognition`）
- `face_orientation.py` 的 `OrientationRecognizer` 结合 SCRFD 检测、关键点定位与 PnP 姿态求解，输出可视化结果、欧拉角以及调节建议（-1 向左、0 居中、1 向右）。
- `modules/` 包含 SCRFD 模型、关键点检测与姿态估计实现；`assets/` 为 ONNX 模型；`utils.py` 提供人脸框修正等辅助函数。

### 手势识别（`src/scripts/gesture_recognition`）
- `gesture_recognition.py` 基于 MediaPipe Tasks，`GestureRecognizer.use(frame)` 返回带注释图像、手势类别与得分。
- `models/gesture_recognizer.task` 为手势模型文件；`utils/visualization_utils.py` 用于绘制手部关键点与批量可视化。

### 指向识别（`src/scripts/pointing_gesture_recognition`）
- `pointing_recognition.py` 的 `PointingRecognizer` 结合 MediaPipe Hands 与自定义分类器识别指向动作，维护关键点历史并返回食指三个关键点坐标。
- `utils/` 提供 FPS 统计、绘制函数与数据采集工具；`model/` 存放基于 TensorFlow 的关键点及轨迹分类器与标签。

### 行李指向示例（`src/scripts/bag.py`）
- 集成 YOLO 检测与 `PointingRecognizer`，结合深度点云（`transformed_depth_point_cloud`、`fx`、`fy`）计算被指向行李的三维位置。

### 姿态识别（`src/scripts/posture_identification`）
- `posture_identification.py` 的 `PostureIdentifier` 使用 MediaPipe Pose Landmarker 与 SVM 分类器（`output/pose_recognizer.pickle`），输出姿态类别与置信度。
- `dataset/`、`models/`、`output/`、`scripts/` 目录分别对应数据素材、模型任务文件、训练产物与预处理脚本。

### YOLO 目标检测（`src/scripts/yolo_detection`）
- `yolo_detection.py` 将 Ultralytics YOLO 封装为 `YoloDetector`，返回带标注图像、是否检测到目标以及结构化 `DetectedObject` 列表。
- `utils/results_utils.py` 定义 `DetectedObject`、Ultralytics 结果转换与统计工具。
- `requirements.txt` 罗列该模块所需依赖，如 `torch`、`supervision`。

### 辅助与演示脚本（`src/scripts`）
- `camera_opencv.py`、`camera_opencv_in_one.py` 示范如何通过键盘切换各模块，并在独立窗口或同一画面展示效果。
- `take_personal_faces.py` 配合 `save_personal_faces` 采集新增人脸数据。
- `test_*.py` 系列脚本用于单模块功能冒烟测试，便于确认驱动与模型状态。
- `split_dataset.py`、`split_yolo_dataset.py` 自动完成 YOLO 数据集拆分，确保随机种子一致。

## 依赖说明
- 每个子模块都提供独立的 `requirements.txt`，可按需安装，避免 ROS 运行环境过于臃肿。
- 常用依赖包括 OpenCV、MediaPipe、Ultralytics YOLO、Scikit-learn、TensorFlow（指向识别分类器）、ONNXRuntime（SCRFD）。
- 摄像头封装需要配合供应商 SDK：Azure Kinect 需安装 `pyk4a`，RealSense 需安装 `pyrealsense2`。

接下来是各模块的详细使用说明与接口文档，请参见相应子目录下的 README 文件(虽然这些文件也不是很详细就是了)

其实以上的模块就是晓萌视觉系统的库,大多是从GitHub上直接扒下来修改得到的,scripts文件夹之外的python文件是测试样本,其具体作用写在各个脚本的目录之中