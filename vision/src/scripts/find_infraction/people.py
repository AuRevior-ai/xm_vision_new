from ultralytics import YOLO
from pyk4a import PyK4A, ColorResolution, Config
import cv2
import  os

class FindPeople:
    def __init__(self, yolo_model_path):
        self.yolo = YOLO(yolo_model_path)
    def use(self, color_image,transformed_depth_point_cloud, fx, fy):
        people_s = []
        if color_image.shape[2] == 4:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        results = self.yolo(color_image)
        crop_image = results[0].plot()
        for result in results:
            for box in result.boxes:
                class_id = box.cls[0]
                xmin, ymin, xmax, ymax = map(float, box.xyxy[0])
                x_center,y_center,_,_ = box.xywh[0]
                if class_id == 0.0:
                    people_s.append({"id": class_id, "position": (xmin, ymin, xmax, ymax),"point":(x_center,y_center)})
        if people_s:
            width = 2600  # 设置图像宽度
            height = 1600  # 设置图像高度
            x1, y1, x2, y2 = map(int, people_s[0]["position"])
            cv2.rectangle(color_image, (x1,y1),(x2,y2), (0,255,0), 2)
            save_path = os.path.join("/home/xiaomeng/桌面", "people.png")
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, color_image)
            else:
                # 先删除现有的 no_drink.png，然后保存
                os.remove(save_path)
                cv2.imwrite(save_path, color_image)
            x_center , y_center = people_s[0]["point"]
            # x_center = (width / 2 / fx) - (1 / fx) * (width / 2 - x_center) if x_center <= width / 2 else (1 / fx) * (x_center - width / 2) + (width / 2 / fx)
            # y_center = (height / 2 / fy) - (1 / fy) * (height / 2 - y_center) if y_center <= height / 2 else (1 / fy) * (y_center - height / 2) + (height / 2 / fy)
            coordinate = transformed_depth_point_cloud[int(y_center)][int(x_center)]
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return '1', coordinate[2]/1000, -coordinate[0]/1000, -coordinate[1]/1000,crop_image
        else:
            return '0', 12000, 12000, 12000, crop_image
    def __del__(self):
        print("已销毁")