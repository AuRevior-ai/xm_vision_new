from ultralytics import YOLO
import cv2
yolo = YOLO("")
def DeskObject(name, color_image, transformed_depth_point_cloud, fx, fy):
    def get_coordinates(class_id):
        for result in results:
            for box in result.boxes:
                if box.cls[0] == class_id:
                    x_center, y_center, _, _ = box.xywh[0]
                    width = 2600  # 设置图像宽度
                    height = 1600  # 设置图像高度
                    x_center = (width / 2 / fx) - (1 / fx) * (width / 2 - x_center) if x_center <= width / 2 else (1 / fx) * (x_center - width / 2) + (width / 2 / fx)
                    y_center = (height / 2 / fy) - (1 / fy) * (height / 2 - y_center) if y_center <= height / 2 else (1 / fy) * (y_center - height / 2) + (height / 2 / fy)
                    coordinate = transformed_depth_point_cloud[int(y_center)][int(x_center)]
                    return coordinate[2] / 1000, -coordinate[0] / 1000, -coordinate[1] / 1000
        return None

    if color_image.shape[2] == 4:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    results = yolo(color_image)

    # 尝试找到指定的物品
    coordinates = get_coordinates(name)
    if coordinates:
        return name, coordinates[0], coordinates[1], coordinates[2]

    # 按顺序查找牛奶，麦片，碗
    for item in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        coordinates = get_coordinates(item)
        if coordinates:
            return item,coordinates[0], coordinates[1], coordinates[2]

    # 如果所有物品都找不到，返回默认坐标
    return 0, 12000, 12000, 12000