#此脚本用于测试体态识别功能
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