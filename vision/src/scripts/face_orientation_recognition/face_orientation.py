"""
This module contains the `OrientationRecognizer` class that can be used to
recognize the orientation of a face relative to the camera.

该模块包含`OrientationRecognizer`类，可用于识别人脸相对于摄像头的朝向。
"""
import os
import cv2
import numpy as np
#检测人脸
from .modules.face_detection import FaceDetector
#面部标志检测
from .modules.mark_detection import MarkDetector
#根据面部标志估计头部姿态
from .modules.pose_estimation import PoseEstimator
from .utils import refine


class OrientationRecognizer:
    """
    A class that can be used to recognize the orientation of a face relative
    to the camera.
    
    一个可用于识别人脸相对于摄像头的朝向的类。
    """
    def __init__(self, frame_width, frame_height):
        """
        Initialize the `OrientationRecognizer` object with the given
        `frame_width` and `frame_height`.

        使用给定的`frame_width`和`frame_height`初始化`OrientationRecognizer`对象。

        Args:
            frame_width: The width of the frame.
            frame_height: The height of the frame.
        """
        # 获取当前脚本文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Setup a face detector to detect human faces.
        self.face_detector = FaceDetector(os.path.join(current_dir, "assets", "face_detector.onnx"))

        # Setup a mark detector to detect landmarks.
        self.mark_detector = MarkDetector(os.path.join(current_dir, "assets", "face_landmarks.onnx"))

        # Setup a pose estimator to solve pose.
        self.pose_estimator = PoseEstimator(frame_width, frame_height)

        self.frame_width = frame_width
        self.frame_height = frame_height

    def use(self, frame):
        """
        Recognize the orientation of a face relative to the camera from the
        given `frame`.

        从给定的`frame`中识别人脸相对于摄像头的朝向。

        Args:
            frame: The frame to recognize the orientation of a face from.

        Returns:
            A tuple containing the following elements:
            - The frame with the orientation of the face annotated.
            - A boolean indicating if a valid face was found in the frame.
            - The angle of deviation of the face relative to the camera.
            - The adjustment based on the angle of deviation of the face
              relative to the camera. It can be one of the following:
              1: Turn right.
              -1: Turn left.
              0: No adjustment needed.

            包含以下元素的元组：
            - 带有人脸朝向的帧。
            - 一个布尔值，指示在帧中是否找到了有效的人脸。
            - 人脸相对于摄像头的朝向偏差的角度。
            - 根据人脸相对于摄像头的朝向偏差的角度获取调整方式。它可以是以下之一：
              1: 向右转。
              -1: 向左转。
              0: 不需要调整。
        """
        face_detector = self.face_detector
        mark_detector = self.mark_detector
        pose_estimator = self.pose_estimator
        frame_width = self.frame_width
        frame_height = self.frame_height

        # Make a copy of the frame to draw on.
        # 复制一份帧以便绘制
        frame = frame.copy()

        # Step 1: Get faces from current frame.
        faces, _ = face_detector.detect(frame, 0.7)#threshold越大检测越准确

        angle = None
        adjustment = 0

        # Any valid face found?
        if len(faces) > 0:
            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector. Note only the first face will be used for
            # demonstration.
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            # Run the mark detection.
            marks = mark_detector.detect([patch])[0].reshape([68, 2])

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Step 3: Try pose estimation with 68 points.
            pose = pose_estimator.solve(marks)

            # Get the angle of deviation of the face relative to the camera,
            # and get the adjustment based on the angle.
            # 获取人脸相对于摄像头的朝向偏差的角度，并根据角度获取调整方式
            angle = get_orientation_angle(pose)
            adjustment = get_adjustment(angle)

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.visualize(frame, pose, color=(0, 255, 0))

            # Do you want to see the axes?
            pose_estimator.draw_axes(frame, pose)

            # Do you want to see the marks?
            # mark_detector.visualize(frame, marks, color=(0, 255, 0))

            # Do you want to see the face bounding boxes?
            # face_detector.visualize(frame, faces)
        
        return frame, len(faces) > 0, angle, adjustment


def get_orientation_angle(pose):
    """
    Get the angle of deviation of the face relative to the camera from the 
    `pose` returned by `pose_estimator.solve()`.

    从`pose_estimator.solve()`返回的`pose`中获取人脸相对于摄像头的朝向偏差的角度.

    Args:
        pose: The pose returned by `pose_estimator.solve()`.

    Returns:
        The angle of deviation of the face relative to the camera.
    """
    rotation_vector = pose[0]

    # Convert the rotation vector to a rotation matrix.
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Create a 3x4 matrix that contains the rotation matrix and a 3x1 zero vector as the fourth column.
    # 创建一个3x4的矩阵，其中包含旋转矩阵和一个3x1的零向量作为第四列
    projection_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))

    # Calculate the Euler angles from the rotation matrix.
    # 从旋转矩阵计算欧拉角
    _, angle, _ = cv2.decomposeProjectionMatrix(projection_matrix)[-1]

    return angle

def get_adjustment(angle):
    """
    Get the adjustment based on the angle of deviation of the face 
    relative to the camera.
    
    根据人脸相对于摄像头的朝向偏差的角度获取调整方式
    
    Args:
        angle: The angle of deviation of the face relative to the
        camera.
          
    Returns:
        The adjustment based on the angle of deviation of the face
        relative to the camera. It can be one of the following:
        1: Turn right.
        -1: Turn left.
        0: No adjustment needed."""
    # define the angle threshold in degrees
    # 定义角度阈值，单位为度
    angle_threshold = 7

    if angle > angle_threshold:
        return 1
    elif angle < -angle_threshold:
        return -1
    else:
        return 0
    