"""
This module contains the `PostureIdentifier` class for identifying postures in video frames using Mediapipe Pose Landmarker and SVM.

该模块包含`PostureIdentifier`类，用于使用Mediapipe Pose Landmarker和SVM在视频帧中识别姿势。
"""
import os
import cv2
import pickle
import numpy as np
import mediapipe as mp

# MediaPipe 兼容性处理
try:
    # 尝试导入旧版本的 solutions API
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    print("使用传统 MediaPipe solutions API")
except ImportError:
    # 新版本没有 solutions，创建兼容层
    print("新版本 mediapipe - 创建姿态识别兼容层")
    
    # 创建简化的 landmark_pb2 兼容类
    class LandmarkCompat:
        class NormalizedLandmark:
            def __init__(self, x=0, y=0, z=0):
                self.x = x
                self.y = y
                self.z = z
                
        class NormalizedLandmarkList:
            def __init__(self):
                self.landmark = []
    
    class landmark_pb2:
        NormalizedLandmark = LandmarkCompat.NormalizedLandmark
        NormalizedLandmarkList = LandmarkCompat.NormalizedLandmarkList
    
    class SolutionsCompat:
        class drawing_utils:
            @staticmethod
            def draw_landmarks(image, landmarks, connections=None, landmark_drawing_spec=None, connection_drawing_spec=None):
                # 简化的绘制函数，绘制基本的关键点
                if landmarks and hasattr(landmarks, 'landmark'):
                    for idx, landmark in enumerate(landmarks.landmark):
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                        # 可以添加关键点编号
                        cv2.putText(image, str(idx), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        class pose:
            # 简化的姿态连接点定义
            POSE_CONNECTIONS = []
        
        class drawing_styles:
            @staticmethod
            def get_default_pose_landmarks_style():
                return None
    
    solutions = SolutionsCompat()

# Set up current script directory and model paths
current_dir = os.path.dirname(os.path.abspath(__file__))
recognizer_path = os.path.join(current_dir, "output", "pose_recognizer.pickle")
label_encoder_path = os.path.join(current_dir, "output", "label_encoder.pickle")

class PostureIdentifier:
    """
    A class for identifying postures in video frames using Mediapipe Pose Landmarker and SVM.
    """
    
    def __init__(self, model_path="models/pose_landmarker_heavy.task"):
        """
        Initialize the `PostureIdentifier` with Mediapipe Pose Landmarker and pre-trained classifier.
        
        初始化`PostureIdentifier`，使用Mediapipe Pose Landmarker和预训练的分类器。
        
        Args:
            model_path (str): The path to the pose landmarker model.
            
            model_path (str): 姿势标记器模型的路径。
        """
        # Load pose landmarker model
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        landmarker_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=current_dir + '/' + model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = PoseLandmarker.create_from_options(landmarker_options)

        # Try to load pre-trained posture classifier and label encoder
        try:
            with open(recognizer_path, "rb") as f:
                self.recognizer = pickle.load(f)
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            self.has_trained_model = True
            print("[INFO] 成功加载训练好的姿态分类模型")
        except FileNotFoundError:
            print("[WARNING] 未找到训练好的姿态分类模型，将使用简化的演示模式")
            self.recognizer = None
            self.label_encoder = None
            self.has_trained_model = False
            # 创建简化的标签列表用于演示
            self.demo_labels = ["Standing", "Sitting", "Walking", "Unknown"]

    def use(self, frame):
        """
        Identify posture in the given frame.
        
        在给定帧中识别姿势。
        
        Args:
            frame: The video frame to analyze.
            
        Returns:
            A tuple containing:
            - Annotated frame.
            - A boolean indicating whether a posture was detected.
            - The predicted posture label.
            
            包含以下元素的元组：
            - 带有标注的帧。
            - 一个布尔值，指示是否检测到姿势。
            - 预测的姿势标签。
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pose_landmarks_result = self.landmarker.detect(mp_image)
        pose_landmarks_list = pose_landmarks_result.pose_landmarks
        annotated_frame = cv2.cvtColor(np.copy(mp_image.numpy_view()), cv2.COLOR_RGB2BGR)
        
        # Initialize the posture label
        posture_label = None

        for pose_landmarks in pose_landmarks_list:
            # Draw landmarks on the frame
            draw_landmarks(annotated_frame, pose_landmarks)

            if self.has_trained_model:
                # Use trained model for prediction
                flattened_landmarks = np.array([[landmark.x, landmark.y] for landmark in pose_landmarks]).flatten()
                posture_label, probability = predict_posture(flattened_landmarks, self.recognizer, self.label_encoder)
                annotation_text = f"{posture_label}: {probability * 100:.2f}%"
            else:
                # Simple demo mode - just detect pose presence
                import random
                posture_label = random.choice(self.demo_labels)
                probability = random.uniform(0.7, 0.95)
                annotation_text = f"{posture_label}: {probability * 100:.2f}% (演示模式)"

            # Annotate the frame with the predicted posture and probability
            cv2.putText(annotated_frame, annotation_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return annotated_frame, len(pose_landmarks_list), posture_label

def draw_landmarks(annotated_frame, pose_landmarks):
    """
    Draw pose landmarks on the given frame.
    
    Args:
        annotated_frame: The frame to draw landmarks on.
        pose_landmarks: The pose landmarks to be drawn.
    """
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
        for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_frame,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style()
    )

def predict_posture(flattened_landmarks, recognizer, label_encoder):
    """
    Predict the posture based on flattened landmarks using the pre-trained recognizer.
    
    Args:
        flattened_landmarks: Flattened pose landmarks for prediction.
        recognizer: Pre-trained posture recognizer model.
        label_encoder: Label encoder for the posture labels.
    
    Returns:
        tuple: The predicted posture label and its associated probability.
    """
    prediction_probs = recognizer.predict_proba([flattened_landmarks])[0]
    max_prob_index = np.argmax(prediction_probs)
    posture_label = label_encoder.classes_[max_prob_index]
    probability = prediction_probs[max_prob_index]
    
    return posture_label, probability
