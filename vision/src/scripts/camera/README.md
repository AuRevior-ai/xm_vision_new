# camera

`以下内容借助Github Copilot生成(仅供参考)。`

这个包提供了统一的相机接口，支持多种类型的相机，包括标准的 OpenCV 摄像头、Intel RealSense 深度相机和 Microsoft Azure Kinect 相机。封装了各种相机的基本功能，通过统一的 API 接口，便于在不同相机之间切换使用。

## 功能特点

- 支持多种相机类型：OpenCV摄像头、RealSense深度相机、Azure Kinect相机
- 统一的接口设计，便于相机类型的切换
- 提供彩色图像和深度图像的获取功能（对于支持深度的相机）
- 简单易用的工厂函数，通过相机类型字符串创建相机实例

## 安装依赖

要使用此模块，根据所需的相机类型安装相应依赖：

- `opencv-python`
- `numpy`
- `pyrealsense2`
- `pyk4a`

## 使用方法示例

### 使用工厂函数创建相机

```python
import cv2
from camera import create_camera

# 创建OpenCV相机（使用默认摄像头）
camera = create_camera(0)  # 或者使用具体的摄像头索引

# 创建RealSense相机
# camera = create_camera("realsense")

# 创建Azure Kinect相机
# camera = create_camera("azure_kinect")

# 启动相机
camera.start()

try:
    while True:
        # 获取彩色图像帧
        ret, frame = camera.get_color_frame()
        
        if not ret:
            print("无法获取图像帧")
            break
            
        # 显示图像
        cv2.imshow("Camera Feed", frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 停止相机
    camera.stop()
    cv2.destroyAllWindows()
```

### 获取深度图像（适用于RealSense和Azure Kinect）

```python
import cv2
import numpy as np
from camera import create_camera

# 创建支持深度的相机
camera = create_camera("realsense")  # 或 "azure_kinect"
camera.start()

try:
    while True:
        # 获取彩色图像和深度图像
        _, color_frame = camera.get_color_frame()
        ret_depth, depth_frame = camera.get_depth_frame()
        
        if ret_depth:
            # 可视化深度图像（根据实际需求进行调整）
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # 显示彩色图像和深度图像
            cv2.imshow("Color", color_frame)
            cv2.imshow("Depth", depth_colormap)
        else:
            cv2.imshow("Color", color_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.stop()
    cv2.destroyAllWindows()
```

## API 参考

### 工厂函数

- `create_camera(camera_type)`: 根据指定的类型创建相机实例
  - `camera_type`: 可以是整数（OpenCV相机索引）、"realsense"或"azure_kinect"

### 相机类

所有相机类都实现了相同的接口：

#### `OpenCVCamera`

使用OpenCV访问常规USB摄像头。

```python
camera = OpenCVCamera(camera_id=0)  # 默认使用设备索引0
```

#### `RealSenseCamera`

使用Intel RealSense SDK访问RealSense深度相机。

```python
camera = RealSenseCamera()
```

#### `AzureKinectCamera`

使用pyk4a访问Microsoft Azure Kinect相机。

```python
camera = AzureKinectCamera()
```

### 共同方法

- `start()`: 启动相机
- `capture()`: 捕获一帧（内部使用）
- `get_color_frame()`: 获取彩色图像帧，返回 (success, frame)
- `get_depth_frame()`: 获取深度图像帧，返回 (success, frame)（不支持深度的相机将返回 (False, None)）
- `stop()`: 停止相机并释放资源
- `get_height_and_width()`: 获取图像的高度和宽度

## 注意事项

1. 使用RealSense相机前，确保已正确安装`pyrealsense2`库
2. 使用Azure Kinect相机前，确保已正确安装`pyk4a`库及其依赖
3. 深度图像格式和单位在不同相机之间可能有所不同：
   - RealSense: 16位深度图，单位为毫米
   - Azure Kinect: 16位深度图，单位为毫米
4. 确保在程序结束时调用`stop()`方法释放相机资源

## 相机分辨率

- OpenCVCamera: 根据实际相机而定
- RealSenseCamera: 640x480（彩色和深度）
- AzureKinectCamera: 1280x720（彩色）