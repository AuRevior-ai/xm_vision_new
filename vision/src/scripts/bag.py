#这个文件用于行李箱检测与定位
#详细来说,这个脚本使用YOLO模型检测图像中的行李箱,并结合手指指向识别来确定用户指向的行李箱位置,最终返回行李箱在三维空间中的坐标
import cv2
import numpy as np
from pointing_gesture_recognition.pointing_recognition import PointingRecognizer
from ultralytics import YOLO
class PointLuggage:
    def __init__(self, yolo_model_path):
        self.estimator = PointingRecognizer()
        self.yolo = YOLO(yolo_model_path)
    def use(self, color_image, transformed_depth_point_cloud, fx, fy):
        if color_image.shape[2] == 4:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        results = self.yolo(color_image)
        #print(results[0].names)
        crop_frame = results[0].plot()
        bags = []
        for result in results:
            for box in result.boxes:
                class_name = result.boxes.cls.tolist()
                for i in class_name:
                    if i == 39.0:
                        xmin, ymin, xmax, ymax = map(float, box.xyxy[0])
                        bags.append({"id": i, "position": (xmin, ymin, xmax, ymax)})
        annotated_frame, detect_pointing, index_finger_points = self.estimator.use(color_image)
        if index_finger_points is not None:
            pointing_point = index_finger_points[0]
            pointing_start = index_finger_points[2]
            bag_position = self.detect_bag_plus(pointing_point,pointing_start ,bags)
            if bag_position is not None:
                #coordinate = [0.0, 0.0, 0.0]
                x1, y1, x2, y2 = map(int, bag_position)
                width = 2600  # 设置图像宽度
                height = 1600  # 设置图像高度
                x_center = (x1 + x2) / 2
                x_center = (width / 2 / fx) - (1 / fx) * (width / 2 - x_center) if x_center <= width / 2 else (1 / fx) * (x_center - width / 2) + (width / 2 / fx)
                y_center = (y1 + y2) / 2
                y_center = (height / 2 / fy) - (1 / fy) * (height / 2 - y_center) if y_center <= height / 2 else (1 / fy) * (y_center - height / 2) + (height / 2 / fy)
                coordinate = transformed_depth_point_cloud[int(y_center)][int(x_center)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Bag at ({(x1 + x2) / 2}, {(y1 + y2) / 2})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                '''
                x 代表水平方向的坐标（左右）。
                y 代表垂直方向的坐标（上下）。
                z 代表深度方向的坐标（前后）
                '''
                return coordinate[2]/1000, -coordinate[0]/1000, -coordinate[1]/1000
        #return crop_frame,annotated_frame,2,3,4
    def detect_bag(self, point, bags):
        for bag in bags:
            x1, y1, x2, y2 = bag["position"]
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                return bag["position"]
        return None
    def detect_bag_plus(self, point,point_start, bags):
        max_cos_sim = -1
        dir_x = point[0] - point_start[0]
        dir_y = point[1] - point_start[1]
        for bag in bags:
            x1, y1, x2, y2 = bag["position"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            vec_to_bag = (cx - point[0], cy - point[1])
            dot_product = dir_x * vec_to_bag[0] + dir_y * vec_to_bag[1]
            len_dir = (dir_x ** 2 + dir_y ** 2) ** 0.5
            len_vec = (vec_to_bag[0] ** 2 + vec_to_bag[1] ** 2) ** 0.5

            if len_dir * len_vec == 0:
                continue
            cos_sim = dot_product / (len_dir * len_vec)

            if cos_sim > max_cos_sim and cos_sim > 0.8:  # 阈值可调
                max_cos_sim = cos_sim
                return bag["position"]
    def __del__(self):
        print("已销毁")

# 初始化摄像头
