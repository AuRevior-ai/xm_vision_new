import cv2
# 这个文件是 OpenCV 相机的封装，定义了类 OpenCVCamera 用于初始化、启动、捕获图像和停止相机等等操作
# 实际上,由于几乎不会使用 OpenCV 相机,所以这个文件写的比较简陋

class OpenCVCamera():#类
    def __init__(self, camera_id=0):
        self.camera_id = camera_id#属性camera_id,默认值为0

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)#属性cap，使用cv2.VideoCapture初始化相机

    def capture(self):#捕获图像的方法
        pass
    
    def get_color_frame(self) -> tuple[bool, cv2.typing.MatLike]:#得到彩色帧
        return self.cap.read()

    def get_depth_frame(self)-> tuple[bool, None]:#得到深度帧
        return False, None

    def stop(self):#停止相机的方法
        self.cap.release()
        
    def get_height_and_width(self):#固定的高度
        return 720, 1280
