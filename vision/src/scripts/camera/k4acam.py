# import cv2
# 这个文件是 Azure Kinect 相机的封装，定义了类 AzureKinectCamera 用于初始化、启动、捕获图像和停止相机等等操作
import numpy as np

# 尝试导入真正的 pyk4a，如果失败则使用模拟版本
try:
    import pyk4a
    from pyk4a import Config, PyK4A
    from pyk4a.capture import PyK4ACapture
    print("使用真正的 pyk4a")
    USING_REAL_K4A = True
except ImportError:
    print("pyk4a 未安装，使用模拟版本")
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    from mock_pyk4a import Config, PyK4A, ColorResolution, DepthMode
    import mock_pyk4a as pyk4a
    PyK4ACapture = None
    USING_REAL_K4A = False


class AzureKinectCamera():
    def __init__(self, config_preset="balanced", custom_config=None):
        """
        初始化 Azure Kinect 相机
        
        参数:
        config_preset: 预设配置 ("balanced", "high_quality", "fast", "wide_range")
        custom_config: 自定义配置字典，会覆盖预设
        """
        
        # 预设配置
        presets = {
            "balanced": {
                "color_resolution": pyk4a.ColorResolution.RES_720P,
                "depth_mode": pyk4a.DepthMode.WFOV_2X2BINNED,
                "synchronized_images_only": True,
            },
            "high_quality": {
                "color_resolution": pyk4a.ColorResolution.RES_1080P,
                "depth_mode": pyk4a.DepthMode.NFOV_UNBINNED,
                "synchronized_images_only": True,
            },
            "fast": {
                "color_resolution": pyk4a.ColorResolution.RES_720P,
                "depth_mode": pyk4a.DepthMode.NFOV_2X2BINNED,
                "synchronized_images_only": True,
            },
            "wide_range": {
                "color_resolution": pyk4a.ColorResolution.RES_1080P,
                "depth_mode": pyk4a.DepthMode.WFOV_UNBINNED,
                "synchronized_images_only": True,
            }
        }
        
        # 选择配置
        if config_preset in presets:
            config_dict = presets[config_preset].copy()
            print(f"使用预设配置: {config_preset}")
        else:
            config_dict = presets["balanced"].copy()
            print(f"未知预设 '{config_preset}'，使用 balanced 配置")
        
        # 应用自定义配置
        if custom_config:
            config_dict.update(custom_config)
            print("应用自定义配置")
        
        print("相机配置:")
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
            
        self.k4a = PyK4A(Config(**config_dict))

    def start(self):
        self.k4a.start()

    def capture(self):  # 移除类型注释以避免错误
        return self.k4a.get_capture()
    
    def get_color_frame(self) -> tuple[bool, np.ndarray]:#获取彩色图像帧的方法，返回元组
        capture = self.capture()
        return np.any(capture.color), capture.color[:, :, :3]#这里capture.color[:, :, :3]的意思是获取彩色图像的前三个通道，即RGB通道，忽略可能存在的第四个通道（如Alpha通道）
    
    def get_depth_frame(self) -> tuple[bool, np.ndarray]:#获取深度图像帧的方法，返回元组
        capture = self.capture()
        return np.any(capture.depth), capture.depth

    def stop(self):
        self.k4a.stop()
        
    def get_height_and_width(self):#获取图像的高度和宽度
        return 720, 1280