#此脚本用于检测空椅子并返回其坐标，坐标格式为（z，x，y），单位为米，三维坐标
#定义类，在core中作为服务的其中一个任务使用
import cv2
from ultralytics import YOLO

class EmptyChair:
    def __init__(self, yolo_model_path):
        self.yolo = YOLO(yolo_model_path)
    def use(self, color_image,transformed_depth_point_cloud, fx, fy):
        if color_image.shape[2] == 4:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        results = self.yolo(color_image)
        crop_image = results[0].plot()
        chairs = []
        people = []
        for result in results:
            for box in result.boxes:
                class_id = box.cls[0]
                xmin, ymin, xmax, ymax = map(float, box.xyxy[0])
                if class_id == 56.0:
                    chairs.append({"id": class_id, "position": (xmin, ymin, xmax, ymax)})
                elif class_id == 0.0:
                    people.append({"id": class_id, "position": (xmin, ymin, xmax, ymax)})
        empty_chair_position = self.detect_empty_chair(chairs, people)
        if empty_chair_position is not None:
            width = 2600  # 设置图像宽度
            height = 1600  # 设置图像高度
            x1, y1, x2, y2 = map(int, empty_chair_position)
            x_center = (x1 + x2) / 2
            x_center = (width / 2 / fx) - (1 / fx) * (width / 2 - x_center) if x_center <= width / 2 else (1 / fx) * (x_center - width / 2) + (width / 2 / fx)
            y_center = (y1 + y2) / 2
            y_center = (height / 2 / fy) - (1 / fy) * (height / 2 - y_center) if y_center <= height / 2 else (1 / fy) * (y_center - height / 2) + (height / 2 / fy)
            coordinate = transformed_depth_point_cloud[int(y_center)][int(x_center)]
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, "Empty chair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(color_image, f"Empty chair at {empty_chair_position}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return coordinate[2]/1000, -coordinate[0]/1000, -coordinate[1]/1000

    def detect_empty_chair(self, chairs, people):
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
    def __del__(self):
        print("已销毁")


