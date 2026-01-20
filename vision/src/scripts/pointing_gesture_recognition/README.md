# pointing_gesture_recognition

Pointing gesture recognition package of vision module

这个包是用来识别人的指向手势，基于 [Kazuhito00/hand-gesture-recognition-using-mediapipe (at 19311a68b16b42d3ee89e505e3f185693dbd7147)](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe/tree/19311a68b16b42d3ee89e505e3f185693dbd7147) 仓库的代码进行了修改，封装了`PointingRecognizer`类，能够在实时视频流中(传入视频帧`frame`)识别人的指向手势。

## 使用方法示例

使用OpenCV

```python
import cv2
from pointing_gesture_recognition.pointing_recognition import PointingRecognizer

cap = cv2.VideoCapture(0)

estimator = PointingRecognizer()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pointing_detected_frame, detect_pointing, index_finger_points = estimator.use(frame)

    cv2.imshow('frame', pointing_detected_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 其他说明

`PointingRecognizer`类中手的关键点`landmark_list`与索引值的对应关系如下，可下图(图片来源：[mediapipe手势识别官方说明文档](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer?hl=zh-cn))

![hand_landmarks](readme_assets/hand-landmarks.png)
