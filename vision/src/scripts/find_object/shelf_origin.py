from ultralytics import YOLO
import cv2
import os
yolo = YOLO("/home/xiaomeng/catkin_ws/src/xm_vision/src/scripts/find_object/shelf.pt")
'''
0.467721 0.885595 0.164032 0.135266
0.467227 0.743303 0.176877 0.145806
0.467721 0.521959 0.199605 0.258235
'''
my_dict = {
    'food':['biscuit','chip','lays','bread','maipian','pocky'],
    'daily_use':['handwash','dishsoap','cereal bowl','body wash','spoon'],
    'drink':['water','sprite','cola','orange juice','milk']
}

x_center = [0.486495, 0.494401]
y_center = [0.403382, 0.749451]
width = [0.375494, 0.343874]
height = [0.389987, 0.259991]
xmin1 = int((x_center[0] - width[0]/2)*1920)
xmax1 = int((x_center[0] + width[0]/2)*1920)
ymin1 = int((y_center[0] - height[0]/2)*1080)
ymax1 = int((y_center[0] + height[0]/2)*1080)
xmin2 = int((x_center[1] - width[1]/2)*1920)
xmax2 = int((x_center[1] + width[1]/2)*1920)
ymin2 = int((y_center[1] - height[1]/2)*1080)
ymax2 = int((y_center[1] + height[1]/2)*1080)
def ShelfObjiectDetection(name,color_image, transformed_depth_point_cloud, fx, fy):
    shelf_s = []
    if color_image.shape[2] == 4:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

    frame1 = color_image[ymin1:ymax1, xmin1:xmax1]
    flag1 = findobject(frame1,name)
    if flag1:
        cv2.rectangle(color_image, (xmin1, ymin1), (xmax1, ymax1), (0, 255, 0), 2)
        # 保存图像，图像名字是putname的内容，图片格式是png
        save_path = os.path.join("/home/xiaomeng/桌面", f"{name}.png")
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, color_image)
        else:
            # 先删除现在的annotated_frame.png,然后保存
            os.remove(save_path)
            cv2.imwrite(save_path, color_image)
        return 1
    frame2 = color_image[ymin2:ymax2, xmin2:xmax2]
    flag2 = findobject(frame2, name)
    if flag2:
        cv2.rectangle(color_image, (xmin1, ymin1), (xmax1, ymax1), (0, 255, 0), 2)
        # 保存图像，图像名字是putname的内容，图片格式是png
        save_path = os.path.join("vision", f"{name}.png")
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, color_image)
        else:
            # 先删除现在的annotated_frame.png,然后保存
            os.remove(save_path)
            cv2.imwrite(save_path, color_image)
        return 0
    return 2


def findobject(frame,name):
    results = yolo(frame)
    value = my_dict.get(name)
    for result in results:
        for box in result.boxes:
            class_id = box.cls[0]
            if class_id in value:
                return 1
    return 0
    # for result in results:
    #     for box in result.boxes:
    #         class_id = box.cls[0]
    #         if class_id == name:
    #             x_center, y_center, _, _ = box.xywh[0]
    #             width = 2600  # 设置图像宽度
    #             height = 1600  # 设置图像高度
    #             x_center = (width / 2 / fx) - (1 / fx) * (width / 2 - x_center) if x_center <= width / 2 else (1 / fx) * (x_center - width / 2) + (width / 2 / fx)
    #             y_center = (height / 2 / fy) - (1 / fy) * (height / 2 - y_center) if y_center <= height / 2 else (1 / fy) * (y_center - height / 2) + (height / 2 / fy)
    #             coordinate = transformed_depth_point_cloud[int(y_center)][int(x_center)]
    #             if -coordinate[1]/1000 < 0.2:
    #                 return 1, 1
    #             elif -coordinate[1]/1000 > 0.2 and -coordinate[1]/1000 < 0.4:
    #                 return 1, 2
    #             else:
    #                 return 1, 3
    #         else:
    #             return 0, 0