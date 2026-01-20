# gesture_recognition

Gesture recognition package of vision module

这个包是用来识别人的手势，基于 [MediaPipe 的手势识别功能](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer?hl=zh-cn) 进行了封装，封装了`GestureRecognizer`类，能够在实时视频流中(传入视频帧`frame`)识别人的手势。

## 使用方法示例

使用OpenCV

```python
import cv2
from gesture_recognition.gesture_recognition import GestureRecognizer

cap = cv2.VideoCapture(0)

estimator = GestureRecognizer()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame, has_gesture, category_name, score = estimator.use(frame)

    cv2.imshow('frame', annotated_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
```
