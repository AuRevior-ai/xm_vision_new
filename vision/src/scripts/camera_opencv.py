import cv2
import numpy as np

from face_identification.face_identification import FaceIdentifier
from face_orientation_recognition.face_orientation import OrientationRecognizer
from gesture_recognition.gesture_recognition import GestureRecognizer
from pointing_gesture_recognition.pointing_recognition import PointingRecognizer


cap = cv2.VideoCapture(0)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 模块映射
MODULE_MAPPPING = {
    'face_identification': FaceIdentifier(),
    'face_orientation': OrientationRecognizer(width, height),
    'gesture_recognition': GestureRecognizer(),
    'pointing_gesture_recognition': PointingRecognizer()
}
# 模块名称列表
MODULE_NAME_LIST = list(MODULE_MAPPPING.keys())

# 开始默认所有模块都是不开启的，可以通过按键来开启
# 数字键1-4对应模块1-4，空格键用来关闭所有模块，ESC键用来退出程序
# 按一次数字键，对应模块开启，再按一次关闭
module_status = [False for _ in range(len(MODULE_NAME_LIST))]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for i, module in enumerate(MODULE_NAME_LIST):
        if module_status[i]:
            annotated_frame, *_ = MODULE_MAPPPING[module].use(frame)
            # show the frame
            cv2.imshow(module, annotated_frame)

            # show all effects in one window
            # frame, *_ = MODULE_MAP[module].use(frame)

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
        if not module_status[key - ord('1')]:
            cv2.destroyWindow(MODULE_NAME_LIST[key - ord('1')])