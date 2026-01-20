#此脚本用于实现基于Kinect和YOLOv8的跟随行人功能，包含图像处理、目标检测和坐标转换等功能。
import argparse
import sys
import time
import cv2
from pyk4a import PyK4A, ColorResolution, Config
from ultralytics import YOLO

codeCodes = {
    'black': '0;30', 'bright gray': '0;37',
    'blue': '0;34', 'white': '1;37',
    'green': '0;32', 'bright blue': '1;34',
    'cyan': '0;36', 'bright green': '1;32',
    'red': '0;31', 'bright cyan': '1;36',
    'purple': '0;35', 'bright red': '1;31',
    '***': '0;33', 'bright purple': '1;35',
    'grey': '1;30', 'bright yellow': '1;33',
}
original_x_center, original_y_center, original_box_width, original_box_height = 0,0,0,0
def colored(text, color='green'):
    return "\033[" + codeCodes[color] + "m" + text + "\033[0m"

def image_detection_original_no_image(image, model, fx, fy):
    global width, height
    global original_x_center, original_y_center, original_box_width, original_box_height
    results = model(image)
    detections = []
    for result in results:
        for box in result.boxes:
            x_center, y_center, box_width, box_height = box.xywh[0]
            print(x_center,y_center,box_width,box_width)
            # x_center = (width / 2 / fx) - (1 / fx) * (width / 2 - x_center) if x_center <= width / 2 else (1 / fx) * (x_center - width / 2) + (width / 2 / fx)
            # y_center = (height / 2 / fy) - (1 / fy) * (height / 2 - y_center) if y_center <= height / 2 else (1 / fy) * (y_center - height / 2) + (height / 2 / fy)
            # box_width /= fx
            # box_height /= fy
            class_name = box.cls[0]
            if class_name == 0.0:
                original_x_center, original_y_center, original_box_width, original_box_height = box.xywh[0]
                # print(colored("Person Detected", color='green'))
                detections.append(['person', box.conf[0], [x_center, y_center, box_width, box_height]])
    return detections

def filter_detections(detections, depth_point_cloud):
    new_detection = []
    # height, width, _ = depth_point_cloud.shape
    for detection in detections:
    #     y, x = int(detection[2][1]), int(detection[2][0])
    #     if 0 <= y < height and 0 <= x < width and detection[0].lower() == 'person' and sum([(v / 1000) ** 2 for v in depth_point_cloud[y][x]]) ** 0.5 <= 4:
        if detection[0].lower() == 'person' and sum([(v / 1000) ** 2 for v in depth_point_cloud[int(detection[2][1])][int(detection[2][0])]]) ** 0.5 <= 5:
            new_detection.append(detection)
    return new_detection

def which_person(filtered_detections):
    the_person = filtered_detections[0]
    for detection in filtered_detections:
        if detection[2][2] * detection[2][3] > the_person[2][2] * the_person[2][3]:
            the_person = detection
    return the_person

def are_you_sure(old_person, new_person):
    if (old_person[2][2] * old_person[2][3]) / (new_person[2][2] * new_person[2][3]) > 4:
        return False
    return True

def transform_coordinate(coordinate):
    if len(coordinate) == 3:
        return [coordinate[2], -coordinate[0], -coordinate[1]]
    return coordinate

def left_or_right(old_person):
    global width
    if old_person[2][0] > width // 2:
        return 1  # 从右边出去去了
    return 0  # 从左边出去去了

class FollowPeople:
    def __init__(self):
        self.model = YOLO('/home/xiaomeng/桌面/visionpart/yolov8n.pt')
        self.old_person = ''
        self.coordinate = [0.0,0.0,0.0]
    def use(self, fx, fy, color_image, transformed_depth_point_cloud, TorF):
        global original_x_center, original_y_center, original_box_width, original_box_height
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--debug', type=bool, default=TorF)
        args = parser.parse_args()
        is_debug = args.debug
        model = self.model
        old_person = self.old_person
        old_coordinate = self.coordinate
        global width, height
        width = 2600  # 设置图像宽度
        height = 1600  # 设置图像高度
        turning_right = False
        turning_left = False
        start_time = time.time()
        if color_image.shape[2] == 4:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        detections = filter_detections(
            image_detection_original_no_image(color_image, model, fx, fy),
            transformed_depth_point_cloud)
        # detections = image_detection_original_no_image(color_image, model, fx, fy)
        coordinate = [0.0, 0.0, 0.0]
        if is_debug:
            drawn_image = color_image
        if len(detections) > 0:
            target_person = which_person(detections)
            if old_person == '':
                old_person = target_person.copy()
            print(target_person)
            if are_you_sure(old_person, target_person):
                turning_right = turning_left = False
                old_person = target_person.copy()
                coordinate = transformed_depth_point_cloud[int(target_person[2][1])][int(target_person[2][0])]
                if coordinate[0] != 0.0 or coordinate[1] != 0.0 or coordinate[2] != 0.0:
                    old_coordinate = coordinate.copy()
                if is_debug:
                    cv2.rectangle(img=drawn_image, pt1=(
                        round(original_x_center.item() - original_box_width.item() / 2),
                        round(original_y_center.item() - original_box_height.item() / 2)),
                                  pt2=(round(original_x_center.item() + original_box_width.item() / 2),
                                       round(original_y_center.item() + original_box_height.item() / 2)),
                                  color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(drawn_image, 'FPS: {:.2f}'.format((1 / (time.time() - start_time))), (20, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                thickness=2)
                    cv2.putText(drawn_image, 'Press \'Q\' to Exit!', (20, 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                thickness=2)
                    cv2.putText(drawn_image, "{}".format(coordinate),
                                (int(target_person[2][0].item()), int(target_person[2][1].item())),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            else:
                print(colored("Person is Lost", color='purple'))
                if left_or_right(old_person):
                    turning_right = True
                    print(colored("leave from right", color='green'))
                else:
                    turning_left = True
                    print(colored("leave from left", color='blue'))
        else:
            print(colored("---------------No person-----------------", color="purple"))
            if old_person != '':
                if left_or_right(old_person):
                    turning_right = True
                    print(colored("leave from right", color='green'))
                else:
                    turning_left = True
                    print(colored("leave from left", color='blue'))
        if turning_right:
            coordinate = [-12000, -12000, -12000]
        elif turning_left:
            coordinate = [-9000, -9000, -9000]
        elif len(list(coordinate)) == 0 or coordinate[0] == coordinate[1] == coordinate[2] == 0.0:
            print('coordinate = old_coordinate')
            coordinate = old_coordinate
        coordinate = transform_coordinate([v / 1000 for v in coordinate])
        print(colored("transformed coordinate:{}".format(coordinate), color='bright green'))
        if is_debug:
            return drawn_image, coordinate
        else:
            return coordinate
    def __del__(self):
        print("已销毁")