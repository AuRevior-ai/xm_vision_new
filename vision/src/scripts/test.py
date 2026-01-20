#袋子
'''
import cv2
import numpy as np
from pointing_gesture_recognition.pointing_recognition import PointingRecognizer
from ultralytics import YOLO

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 初始化手势识别器
estimator = PointingRecognizer()

# 初始化YOLOv8模型
yolo = YOLO("D:/pycharm/codepython/ultralytics-main/weights/yolov8n.pt")  # 使用预训练的YOLOv8模型
#carry_my_:color_image 深度云 yolo 析构函数stop
def detect_bag(point, bags):
    for bag in bags:
        x1, y1, x2, y2 = bag["position"]
        if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
            return bag["position"]
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO检测袋子
    results = yolo(frame)
    print(results[0].names)
    crop_frame = results[0].plot()
    cv2.imshow('YOLO Detection', crop_frame)
    bags = []
    for result in results:
        for box in result.boxes:
            class_name = result.boxes.cls.tolist()
            for i in class_name:
                if i == 56.0:
                    xmin, ymin, xmax, ymax = map(float, box.xyxy[0])
                    print(xmin, xmax, ymin, ymax)
                    bags.append({"id": i, "position": (xmin, ymin, xmax, ymax)})


    # 使用手势识别器识别手势
    annotated_frame, detect_pointing, index_finger_points = estimator.use(frame)

    if index_finger_points is not None:
        #print(index_finger_points[0])
        # 假设手指指向的点是手势识别结果中的某个关键点
        pointing_point = index_finger_points[0] # 示例点，实际应从手势识别结果中获取
        print(index_finger_points[0])

        # 检测被指向的袋子
        bag_id = detect_bag(pointing_point, bags)
        print(bag_id)
        if bag_id is not None:
            x1, y1, x2, y2 = map(int, bag_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Pointing to bag {bag_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('frame', annotated_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
#空椅子
#empty_chair color_image 深度云 yolo 析构函数

import cv2
from ultralytics import YOLO

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 初始化YOLOv8模型
yolo = YOLO("D:/pycharm/codepython/ultralytics-main/weights/yolov8n.pt")  # 使用预训练的YOLOv8模型

def detect_empty_chair(chairs, people):
    for chair in chairs:
        chair_x1, chair_y1, chair_x2, chair_y2 = chair["position"]
        is_empty = True
        for person in people:
            person_x1, person_y1, person_x2, person_y2 = person["position"]
            if not (person_x2 < chair_x1 or person_x1 > chair_x2 or person_y2 < chair_y1 or person_y1 > chair_y2):
                is_empty = False
                break
        if is_empty:
            return chair["position"]
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO检测椅子和人
    results = yolo(frame)
    chairs = []
    people = []
    for result in results:
        for box in result.boxes:
            class_id = box.cls[0]
            xmin, ymin, xmax, ymax = map(float, box.xyxy[0])
            if class_id == 56.0:  # 椅子的类别ID
                chairs.append({"id": class_id, "position": (xmin, ymin, xmax, ymax)})
            elif class_id == 0.0:  # 人的类别ID
                people.append({"id": class_id, "position": (xmin, ymin, xmax, ymax)})

    # 检测空椅子
    empty_chair_position = detect_empty_chair(chairs, people)
    print(empty_chair_position)
    if empty_chair_position is not None:
        x1, y1, x2, y2 = map(int, empty_chair_position)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Empty chair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Empty chair at {empty_chair_position}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''

def capture_faces(self, name):
    while True:
        capture = self.k4a.get_capture()
        frame = capture.color
        if frame is None:
            break

        # 将图���从 RGBA 转换为 RGB
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        face_identified_frame, faces_amount, faces_boxes = self.estimator.use(frame)

        cv2.imshow('frame', face_identified_frame)

        if faces_amount > 0:
            (h, w) = frame.shape[:2]
            box = faces_boxes[0] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            fH, fW = endY - startY, endX - startX
            startX = int(max(0, startX - fW * self.EXPAND_FACTOR))
            endX = int(min(w, endX + fW * self.EXPAND_FACTOR))
            startY = int(max(0, startY - fH * self.EXPAND_FACTOR))
            endY = int(min(h, endY + fH * self.EXPAND_FACTOR))
            face = frame[startY:endY, startX:endX]
            cv2.imshow('taked_face', face)

            if self.count % self.STRIDE == 0:
                self.taked_faces.append(face)
            self.count += 1

        if len(self.taked_faces) >= self.N:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    save_faces(name, self.taked_faces)
    self.stop_camera()



'''