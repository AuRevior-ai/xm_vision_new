from ultralytics import YOLO
import cv2
class FindPeopleNoShoes:
    def __init__(self, yolo_model_path1, yolo_model_path2):
        self.yolo1 = YOLO(yolo_model_path1)
        self.yolo2 = YOLO(yolo_model_path2)
    def use(self, color_image,transformed_depth_point_cloud, fx, fy):
        people_s = []
        if color_image.shape[2] == 4:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        results = self.yolo1(color_image)
        crop_image = results[0].plot()
        for result in results:
            for box in result.boxes:
                class_id = box.cls[0]
                xmin, ymin, xmax, ymax = map(float, box.xyxy[0])
                if class_id == 0.0:
                    people_s.append({"id": class_id, "position": (xmin, ymin, xmax, ymax)})
        if people_s:
            for people in people_s:
                xmin, ymin, xmax, ymax = people["position"]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                true_image = color_image[ymin:ymax, xmin:xmax]
                results = self.yolo2(true_image)
                true_image = results[0].plot()
                for result in results:
                    for box in result.boxes:
                        shoe_class_id = box.cls[0]
                        if shoe_class_id == 1.0:
                            width = 2600  # 设置图像宽度
                            height = 1600  # 设置图像高度
                            x1, y1, x2, y2 = map(int, people["position"])
                            x_center = (x1 + x2) / 2
                            x_center = (width / 2 / fx) - (1 / fx) * (width / 2 - x_center) if x_center <= width / 2 else (1 / fx) * (x_center - width / 2) + (width / 2 / fx)
                            y_center = (y1 + y2) / 2
                            y_center = (height / 2 / fy) - (1 / fy) * (height / 2 - y_center) if y_center <= height / 2 else (1 / fy) * (y_center - height / 2) + (height / 2 / fy)
                            coordinate = transformed_depth_point_cloud[int(y_center)][int(x_center)]
                            return '1', coordinate[2]/1000, -coordinate[0]/1000, -coordinate[1]/1000
        return '0', 12000, 12000, 12000
    def __del__(self):
        print("已销毁")
