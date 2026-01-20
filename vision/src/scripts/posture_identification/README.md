# posture_identification

Posture identification package of vision module

这个包是用来识别人体姿态的，基于 [MediaPipe 的姿势特征点检测](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=zh-cn) 功能，并参考 [aakashjhawar/face-recognition-using-deep-learning](https://github.com/aakashjhawar/face-recognition-using-deep-learning/tree/6279219e7c9569eea3fa5ce16b5331d495fd4e33) 仓库的代码，使用SVM分类器进行具体姿态的识别。

封装了`PostureIdentifier`类，能够在实时视频流中(传入视频帧`frame`)识别人体姿态，并返回标注了姿态的视频帧。

## 使用方法示例

使用OpenCV

```python
import cv2
from posture_identification.posture_identification import PostureIdentifier

cap = cv2.VideoCapture(0)

estimator = PostureIdentifier()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame, has_posture, posture_label = estimator.use(frame)

    cv2.imshow('frame', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 其他文件说明

-   `preprocess.py`: 预处理，运行`scripts/extract_keypoints.py`和`scripts/train_model.py`文件

## 其他说明

`dataset` 文件夹中用于存放用于训练的数据集，每个子文件夹名为一个类别，每个类别中存放对应的数据集，如：

```
dataset
 ├─Lying
 ├─Squatting
 ├─Standing
 └─Walking
```

数据集图片中尽可能只包含一个人，并且人体在图片中是完整的（从头到脚），因为 MediaPipe 的姿势特征点检测即使图片中的人不完整（不完整部分的）其余特征点也会有检测结果/数据（正常特征点数据x和y都在 `[0, 1]` 区间内，不完整部分的特征点x或y会大于 `1`），可能会使得训练出来的模型在预测时有较大的误差。