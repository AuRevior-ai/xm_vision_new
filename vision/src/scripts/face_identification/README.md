# face_identification

Face identification package of vision module

这个包是用来识别人脸的身份，基于 [aakashjhawar/face-recognition-using-deep-learning (at 6279219e7c9569eea3fa5ce16b5331d495fd4e33)](https://github.com/aakashjhawar/face-recognition-using-deep-learning/tree/6279219e7c9569eea3fa5ce16b5331d495fd4e33) 仓库的代码进行了修改，封装了`FaceIdentifier`类，能够在实时视频流中(传入视频帧`frame`)识别人脸的身份。

## 使用方法

使用OpenCV调用摄像头，实时识别人脸的身份代码示例

```python
import cv2
from face_identification.face_identification import FaceIdentifier

cap = cv2.VideoCapture(0)

estimator = FaceIdentifier()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_identified_frame, faces_amount = estimator.use(frame)

    cv2.imshow('frame', face_identified_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 其他文件说明

-   `save_personal_faces.py`: 保存人脸的视频帧为数据集
-   `preprocess.py`: 预处理，运行`scripts/extract_embeddings.py`和`scripts/train_model.py`文件