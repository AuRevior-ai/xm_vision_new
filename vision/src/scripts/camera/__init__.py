from typing import Union

# 安全导入，如果依赖不满足则跳过
try:
    from .rscam import RealSenseCamera
    REALSENSE_AVAILABLE = True
except ImportError:
    print("警告: pyrealsense2 未安装，RealSense 相机不可用")
    RealSenseCamera = None
    REALSENSE_AVAILABLE = False

try:
    from .k4acam import AzureKinectCamera
    AZURE_KINECT_AVAILABLE = True
except ImportError:
    print("警告: pyk4a 未安装，Azure Kinect 相机不可用")
    AzureKinectCamera = None
    AZURE_KINECT_AVAILABLE = False

from .opencvcam import OpenCVCamera


def create_camera(camera_type: Union[str, int], **kwargs):
    """
    创建相机对象
    
    参数:
    camera_type: 相机类型 ("realsense", "azure_kinect", 或整数表示OpenCV相机ID)
    **kwargs: 传递给相机构造函数的额外参数
    
    对于 Azure Kinect，可用参数:
    - config_preset: "balanced", "high_quality", "fast", "wide_range"
    - custom_config: 自定义配置字典
    """
    if camera_type == "realsense":
        if not REALSENSE_AVAILABLE:
            raise ImportError("RealSense 相机不可用，请安装 pyrealsense2")
        return RealSenseCamera(**kwargs)
    elif camera_type == "azure_kinect":
        if not AZURE_KINECT_AVAILABLE:
            raise ImportError("Azure Kinect 相机不可用，请安装 pyk4a 和 Azure Kinect SDK")
        return AzureKinectCamera(**kwargs)
    elif camera_type is not str:
        return OpenCVCamera(camera_type)
    else:
        raise ValueError("Unknown camera type")
'''
# 初始化子模块
git submodule init

# 更新子模块
git submodule update
'''