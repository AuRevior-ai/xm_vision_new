from ultralytics import YOLO
import cv2
yolo = YOLO("/home/xiaomeng/catkin_ws/src/xm_vision/src/scripts/find_object/breakfast.pt")
my_dict = {
    "drink":[3.0,4.0,11.0],
    "chip":[0.0,1.0,2.0],
    "biscuit":[5.0,6.0,7.0,8.0],
    "laundry detergent":[9.0,10.0]
}
'''
0.467721 0.885595 0.164032 0.135266
0.467227 0.743303 0.176877 0.145806
0.467721 0.521959 0.199605 0.258235
'''
x_center = [0.467721, 0.467227, 0.467721]
y_center = [0.885595, 0.743303, 0.521959]
width = [0.164032, 0.176877, 0.199605]
height = [0.135266, 0.145806, 0.258235]
xmin1 = int((x_center[0] - width[0]/2)*1920)
xmax1 = int((x_center[0] + width[0]/2)*1920)
ymin1 = int((y_center[0] - height[0]/2)*1080)
ymax1 = int((y_center[0] + height[0]/2)*1080)
xmin2 = int((x_center[1] - width[1]/2)*1920)
xmax2 = int((x_center[1] + width[1]/2)*1920)
ymin2 = int((y_center[1] - height[1]/2)*1080)
ymax2 = int((y_center[1] + height[1]/2)*1080)
xmin3 = int((x_center[2] - width[2]/2)*1920)
xmax3 = int((x_center[2] + width[2]/2)*1920)
ymin3 = int((y_center[2] - height[2]/2)*1080)
ymax3 = int((y_center[2] + height[2]/2)*1080)
def ShelfObjectDetection(name,color_image, transformed_depth_point_cloud, fx, fy):
    shelf_s = []
    if color_image.shape[2] == 4:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

    frame1 = color_image[ymin1:ymax1, xmin1:xmax1]
    flag1 = findobject(frame1,name)
    if flag1:
        return 1,frame1
    frame2 = color_image[ymin2:ymax2, xmin2:xmax2]
    flag2 = findobject(frame2, name)
    if flag2:
        return 2,frame2
    frame3 = color_image[ymin3:ymax3, xmin3:xmax3]
    flag3 = findobject(frame3, name)
    if flag3:
        return 3,frame3
    return 4,color_image


def findobject(frame,name):
    values = my_dict.get(name)
    results = yolo(frame)
    frame = results[0].plot()
    for result in results:
        for box in result.boxes:
            class_id = box.cls[0]
            if class_id in values:
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