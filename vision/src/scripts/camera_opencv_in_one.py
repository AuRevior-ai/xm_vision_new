# 这个脚本演示了如何在一个窗口中集成多个视觉模块的功能
import cv2
import numpy as np

from face_identification.face_identification import FaceIdentifier
from face_orientation_recognition.face_orientation import OrientationRecognizer
from gesture_recognition.gesture_recognition import GestureRecognizer
from pointing_gesture_recognition.pointing_recognition import PointingRecognizer

from camera import create_camera


cam = create_camera("azure_kinect")#打开_azure_kinect相机

height, width = cam.get_height_and_width()#获取相机的高度和宽度

MODULE_MAP = {#模型映射字典
    'face_identification': FaceIdentifier(),#人脸识别模块
    'face_orientation': OrientationRecognizer(width, height),#人脸朝向识别模块
    'gesture_recognition': GestureRecognizer(),#手势识别模块
    'pointing_gesture_recognition': PointingRecognizer()#指向手势识别模块
}

MODULE_LIST = [
    'face_identification',
    'face_orientation',
    'gesture_recognition',
    'pointing_gesture_recognition'
]

# 开始默认所有模块都是不开启的，可以通过按键来开启
# 数字键1-4对应模块1-4，空格键用来关闭所有模块，ESC键用来退出程序
# 按一次数字键，对应模块开启，再按一次关闭
module_status = [False for _ in range(4)]#模块状态列表

cam.start()

while True:
    ret, frame = cam.get_color_frame()
    if not ret:
        continue

    for i, module in enumerate(MODULE_LIST):
        if module_status[i]:
            # annotated_frame, *_ = MODULE_MAP[module].use(frame)
            # # show the frame
            # cv2.imshow(module, annotated_frame)

            # show all effects in one window
            frame, *_ = MODULE_MAP[module].use(frame)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
    elif key == ord(' '):
        module_status = [False for _ in range(4)]
        cv2.destroyAllWindows()
    elif ord('1') <= key <= ord('4'):
        module_status[key - ord('1')] = not module_status[key - ord('1')]
        # if the module is turned off, close the window
        # if not module_status[key - ord('1')]:
        #     cv2.destroyWindow(MODULE_LIST[key - ord('1')])