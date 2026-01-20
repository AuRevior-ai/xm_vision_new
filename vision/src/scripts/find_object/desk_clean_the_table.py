from ultralytics import YOLO
import cv2
import os

object_dic_reverse = {
    0.0 : 'plate',
    1.0 : 'fork',
    2.0 : 'spoon',
    3.0 : 'bowl'
}

yolo = YOLO("/home/xiaomeng/catkin_ws/src/xm_vision/src/scripts/find_object/cleanbest.pt")
def DeskObject_clean(name, color_image, transformed_depth_point_cloud, fx, fy, flag):
    print(transformed_depth_point_cloud.shape)
    def get_coordinates(class_id, frame, flag):
        for result in results:
            for box in result.boxes:
                conf = box.conf
                if box.cls[0] == class_id and conf >= 0.55:
                    x_center, y_center, w, h  = box.xywh[0]
                    position = box.xyxy[0]
                    x1,y1,x2,y2 = map(int,position)
                    putname = object_dic_reverse[class_id]

                    width = 2600  # 设置图像宽度
                    height = 1600  # 设置图像高度
                    #x_center = (width / 2 / fx) - (1 / fx) * (width / 2 - x_center) if x_center <= width / 2 else (1 / fx) * (x_center - width / 2) + (width / 2 / fx)
                    #y_center = (height / 2 / fy) - (1 / fy) * (height / 2 - y_center) if y_center <= height / 2 else (1 / fy) * (y_center - height / 2) + (height / 2 / fy)
                    # x_center = x_center*w
                    # y_center = y_center*h
                    x_origin = x_center.item()
                    y_origin = y_center.item()
                    x = int(x_origin)
                    y = int(y_origin)
                    print(x_center,y_center)
                    # if(flag == 1):
                    #    y = int(real_y)  
                    index = y*640 + x
                    print(index)
                    cv2.rectangle(frame, (x1,y1) ,(x2,y2) ,(0,255,0) ,2)
                    cv2.putText(frame,putname, (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX, 0.9 , (0,255,0), 2)
                    save_path = os.path.join('/home/xiaomeng/桌面', f'{putname}.png')
                    if not os.path.exists(save_path):
                        cv2.imwrite(save_path,frame)
                    else:
                        os.remove(save_path)
                        cv2.imwrite(save_path,frame)
                    coordinate = transformed_depth_point_cloud[index]
                    return coordinate[0] , coordinate[1] , coordinate[2] , y_origin
        return None,None,None, 481

    # if color_image.shape[2] == 4:
    #     color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    results = yolo(color_image)
    #color_image = results[0].plot()

    # 尝试找到指定的物品
    coordinates_1,coordinates_2,coordinates_3,y = get_coordinates(name, color_image, flag)
    if coordinates_1:
        return name, coordinates_1, coordinates_2, coordinates_3, y

    # 按顺序查找牛奶，麦片，碗
    

    # 如果所有物品都找不到，返回默认坐标
    return name, 12000, 12000, 12000 ,481