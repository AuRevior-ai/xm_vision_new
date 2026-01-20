from ultralytics import YOLO
import cv2
yolo = YOLO("")
def ShelfObjectDetection(name,color_image, transformed_depth_point_cloud, fx, fy):
    shelf_s = []
    if color_image.shape[2] == 4:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    results = yolo(color_image)
    crop_image = results[0].plot()
    for result in results:
        for box in result.boxes:
            class_id = box.cls[0]
            if class_id == name:
                x_center, y_center, _, _ = box.xywh[0]
                width = 2600  # 设置图像宽度
                height = 1600  # 设置图像高度
                x_center = (width / 2 / fx) - (1 / fx) * (width / 2 - x_center) if x_center <= width / 2 else (1 / fx) * (x_center - width / 2) + (width / 2 / fx)
                y_center = (height / 2 / fy) - (1 / fy) * (height / 2 - y_center) if y_center <= height / 2 else (1 / fy) * (y_center - height / 2) + (height / 2 / fy)
                coordinate = transformed_depth_point_cloud[int(y_center)][int(x_center)]
                if -coordinate[1]/1000 < 0.2:
                    return 1, 1
                elif -coordinate[1]/1000 > 0.2 and -coordinate[1]/1000 < 0.4:
                    return 1, 2
                else:
                    return 1, 3
            else:
                return 0, 0