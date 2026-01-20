# face_orientation_recognition

Face orientation recognition package of vision module

这个包是用来识别人脸相对于相机的朝向(正对相机视为0度，左转为正，右转为负)，基于 [yinguobing/head-pose-estimation (at e0c37703e31807e49ab46d919cd010a7809b2300)](https://github.com/yinguobing/head-pose-estimation/tree/e0c37703e31807e49ab46d919cd010a7809b2300) 仓库的代码进行了修改，封装了`OrientationEstimator`类，能够在实时视频流中(传入视频帧`frame`)识别人脸朝向。

## 使用方法示例

使用OpenCV

```python
import cv2
from face_orientation_recognition.face_orientation import OrientationRecognizer

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

estimator = OrientationRecognizer(frame_width, frame_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    estimated_frame, has_face, angle, adjustment = estimator.use(frame)

    if has_face:
        cv2.imshow('frame', estimated_frame)
        print(f'angle: {angle}, adjustment: {adjustment}')
    else:
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```