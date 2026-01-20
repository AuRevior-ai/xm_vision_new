"""
该模块包含用于识别手势的`GestureRecognizer`类。
"""
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
import cv2

from .utils.visualization_utils import *


# 获取当前脚本文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))


class GestureRecognizer:
    def __init__(self, model_asset_path='models/gesture_recognizer.task'):
        """
        Initializes the `GestureRecognizer` object.
        
        初始化`GestureRecognizer`对象。
        
        Args:
            model_asset_path: The path to the gesture recognizer model asset.
        """
        # STEP 2: Create an GestureRecognizer object.
        base_options = python.BaseOptions(model_asset_path=current_dir + '/' + model_asset_path)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def use(self, frame):
        """
        Recognizes gestures in the input frame.

        在输入帧中识别手势。

        Args:
            frame: The input frame.

        Returns:
            A tuple containing the following elements:
            - The annotated frame.
            - A boolean value indicating whether a gesture is recognized.
            - The category name of the recognized gesture.
            - The score of the recognized gesture.

            包含以下元素的元组：
            - 带标注的帧。
            - 一个布尔值，指示是否识别了手势。
            - 识别的手势的类别名称。
            - 识别的手势的分数。
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # STEP 3: Load the input image.
        # Load the input image from a numpy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # STEP 4: Recognize gestures in the input image.
        recognition_result = self.recognizer.recognize(mp_image)

        if recognition_result.gestures:
            # STEP 5: Process the result. In this case, visualize it.
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks

            annotated_image = draw_one_image_with_gestures_and_hand_landmarks(mp_image, top_gesture, hand_landmarks)

            return annotated_image, True, top_gesture.category_name, top_gesture.score
        else:
            return frame, False, None, None
        #带标注的帧，是否识别了手势，识别的手势的类别名称，识别的手势的分数