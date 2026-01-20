"""新版基于 MediaPipe Tasks API 的指向手势识别。

这里不再使用旧版 mp.solutions.hands + 自训练 KeyPointClassifier，
而是直接复用手势识别模块中已经验证可用的 GestureRecognizer
来获取手部关键点，然后根据食指关键点计算一个“指向方向”。
"""

import os
from collections import Counter, deque

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


from .utils import CvFpsCalc
from .model import KeyPointClassifier, PointHistoryClassifier
from .utils.utils import (
    calc_bounding_rect,
    calc_landmark_list,
    pre_process_landmark,
    pre_process_point_history,
    draw_bounding_rect,
    draw_landmarks,
    draw_point_history,
    draw_info,
    draw_info_text,
)


# 获取当前脚本文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))


class _HandLandmarksWrapper:
    """将 Tasks API 输出的 hand_landmarks(list[Landmark]) 包装成拥有 .landmark 属性的对象，
    以兼容原先 utils.calc_bounding_rect / calc_landmark_list 的接口。
    """

    def __init__(self, lm_list):
        self.landmark = lm_list


class PointingRecognizer:
    """使用 MediaPipe Tasks GestureRecognizer 做指向可视化。

    - 使用与 gesture_recognition 模块相同的 gesture_recognizer.task 模型；
    - 只要检测到手，就绘制手部关键点；
    - 额外根据食指三个关键点计算指向方向。
    """

    def __init__(self, model_asset_path="models/gesture_recognizer.task"):
        # FPS 计算与历史点记录（沿用原来的工具类，便于在图像上显示信息）
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)

        # 创建 GestureRecognizer，与 gesture_recognition.GestureRecognizer 相同风格
        base_options = mp_python.BaseOptions(
            model_asset_path=os.path.join(current_dir, model_asset_path)
        )
        options = mp_vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = mp_vision.GestureRecognizer.create_from_options(options)

        # 加载原来的关键点/轨迹分类器和标签，恢复严格 hand_sign_id==2 的语义
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        # 标签文件与原版保持一致
        with open(
            os.path.join(current_dir, "model/keypoint_classifier/keypoint_classifier_label.csv"),
            encoding="utf-8-sig",
        ) as f:
            import csv

            reader = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in reader]

        with open(
            os.path.join(current_dir, "model/point_history_classifier/point_history_classifier_label.csv"),
            encoding="utf-8-sig",
        ) as f:
            import csv

            reader = csv.reader(f)
            self.point_history_classifier_labels = [row[0] for row in reader]

    def _calc_index_finger_points(self, hand_landmarks, image_shape):
        """从手部关键点中提取食指三个关键点的像素坐标。

        MediaPipe Hands / GestureRecognizer 的关键点索引约定：
        5: 食指根部, 6: 第一关节, 7: 第二关节, 8: 指尖。
        这里我们用 6, 7, 8 三个点，分别作为 base / mid / tip。
        """

        h, w = image_shape[0], image_shape[1]

        def to_pixel(lm):
            return [int(lm.x * w), int(lm.y * h)]

        tip = to_pixel(hand_landmarks[8])
        mid = to_pixel(hand_landmarks[7])
        base = to_pixel(hand_landmarks[6])
        return [tip, mid, base]

    def use(self, frame):
        """在输入帧中识别手部并估计食指指向。

        返回：
        - annotated_frame: 带关键点和简单指向信息的图像；
        - detect_pointing: 只要检测到至少一只手就为 True；
        - index_finger_points: 如果检测到手，则为 [tip, mid, base] 像素坐标列表，否则为 None。
        """

        fps = self.cvFpsCalc.get()

        # MediaPipe Tasks 使用 RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 调用 GestureRecognizer 获取手关键点
        result = self.recognizer.recognize(mp_image)

        # 默认输出
        debug_image = frame.copy()
        detect_pointing = False
        index_finger_points = None

        # 为了“严格版”逻辑，我们尽量复刻原始流程：
        # - 把 hand_landmarks 转为 landmark_list 像素坐标；
        # - pre_process_landmark -> KeyPointClassifier -> hand_sign_id；
        # - 只有 hand_sign_id == 2 时才认为是指向手势，并返回食指三个关键点。

        if result.hand_landmarks:
            for lm_list in result.hand_landmarks:
                wrapped = _HandLandmarksWrapper(lm_list)

                # 计算外接矩形 & 像素级关键点列表（沿用原 utils 实现）
                brect = calc_bounding_rect(debug_image, wrapped)
                landmark_list = calc_landmark_list(debug_image, wrapped)

                # 特征预处理
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, self.point_history
                )

                # 手势分类
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                if hand_sign_id == 2:  # 严格版：仅类 2 视为“指向手势”
                    self.point_history.append(landmark_list[8])  # 食指指尖像素坐标
                    detect_pointing = True
                    index_finger_points = [
                        landmark_list[8],
                        landmark_list[7],
                        landmark_list[6],
                    ]
                else:
                    self.point_history.append([0, 0])

                # 轨迹手势分类（与原版保持一致）
                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(
                        pre_processed_point_history_list
                    )

                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                # 绘制调试信息（骨架 + 文本）
                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    None,  # 新版没有 handedness，这里先不显示左右手信息
                    self.keypoint_classifier_labels[hand_sign_id],
                    self.point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            self.point_history.append([0, 0])

        # 画轨迹与 FPS 信息
        debug_image = draw_point_history(debug_image, self.point_history)
        debug_image = draw_info(debug_image, fps, 0, -1)

        return debug_image, detect_pointing, index_finger_points

