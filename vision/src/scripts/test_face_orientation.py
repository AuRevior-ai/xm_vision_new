#此脚本用于测试人脸朝向识别功能
#通过测试,可以在环境xiaomeng_env下使用
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