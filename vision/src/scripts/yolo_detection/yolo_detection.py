"""
YOLO 目标检测模块。

该模块提供了使用 YOLO (You Only Look Once) 深度学习模型进行目标检测的功能。
可以检测图像中的多个物体，并返回带有标注和分类信息的结果。
"""
from ultralytics import YOLO
import supervision as sv
from ultralytics.engine.results import Results
from .utils import *

class YoloDetector:
    """
    YOLO 目标检测器类。
    
    该类封装了 YOLO 模型的加载和使用，提供了简单的接口来对图像进行目标检测。
    
    Attributes:
        model (YOLO): 加载的 YOLO 模型实例。
    """
    def __init__(self, model):
        """
        初始化 YOLO 检测器。
        
        Args:
            model (str or YOLO): YOLO 模型的路径或已加载的模型实例。
                可以是本地模型文件路径或预训练模型名称。
        """
        self.model = YOLO(model)

    def use(self, frame):
        """
        使用加载的 YOLO 模型对输入帧进行目标检测。
        
        Args:
            frame (numpy.ndarray): 输入的图像帧，BGR 格式。
        
        Returns:
            tuple: 包含以下三个元素的元组:
            - annotated_frame (numpy.ndarray): 标注了检测结果的图像帧
            - is_empty (bool): 是否没有检测到任何目标
            - detected_objects (list): 检测到的目标对象列表，每个元素为 DetectedObject 实例
        """
        results: Results = self.model(frame)[0]
        
        annotated_frame = get_annotated_frame(frame.copy(), results)
        detected_objects = ultralytics_results_to_detected_objects(results)
        return annotated_frame, is_results_empty(results), detected_objects


def get_annotated_frame(frame, results):
    """
    将检测结果标注到原始图像帧上。
    
    使用 supervision 库将检测到的边界框和类别标签绘制在图像上。
    
    Args:
        frame (numpy.ndarray): 原始图像帧。
        results (Results): YOLO 模型的检测结果对象。
    
    Returns:
        numpy.ndarray: 标注了检测结果的图像帧。
    """
    detections = sv.Detections.from_ultralytics(results)
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.7, text_padding=0)
    
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    
    return annotated_frame