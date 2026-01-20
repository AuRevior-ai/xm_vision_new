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
   '0.0': 'food',
   '1.0': 'food',
   '2.0': 'food',
   '3.0': 'food',
   '4.0': 'food',
   '5.0': 'daily_use',
   '6.0': 'daily_use',
   '7.0': 'drink',
   '8.0': 'drink',
   '9.0': 'drink',
   '10.0': 'drink',
   '11.0': 'daily_use',
   '12.0': 'daily_use',
   '13.0': 'daily_use',
   '14.0': 'drink',
   '15.0': 'food'
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
def ShelfObjiectDetection2(color_image):
    shelf_s = []
    if color_image.shape[2] == 4:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

    frame1 = color_image[ymin1:ymax1, xmin1:xmax1]
    classname1 = findobject(frame1)
    frame2 = color_image[ymin2:ymax2, xmin2:xmax2]
    classname2 = findobject(frame2)
    #classname1,classname2在classname中，返回classname剩下的那个元素
    if classname1 == 'food' and classname2 =='drink':
        return 'daily_use'
    elif classname1 == 'food' and classname2 == 'daily_use':
        return 'drink'
    elif classname1 == 'drink' and classname2 == 'food':
        return 'daily_use'
    elif classname1 == 'drink' and classname2 == 'daily_use':
        return 'food'
    elif classname1 == 'daily_use' and classname2 == 'food':
        return 'drink'
    elif classname1 == 'daily_use' and classname2 == 'drink':
        return 'food'
    else:
        return 'no_class'





def findobject(frame):
    class_list = []
    results = yolo(frame)
    for result in results:
        for box in result.boxes:
            class_id = box.cls[0]
            class_id = str(class_id)
            classname = my_dict.get(class_id)
            class_list.append(classname)
    if class_list == []:
        return 'no_class'
    else:
    #返回class_list中出现次数最多的元素
        return max(set(class_list), key=class_list.count)

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