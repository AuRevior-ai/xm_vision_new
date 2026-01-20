"""
YOLO 检测结果处理工具模块。

该模块提供了用于处理和转换 YOLO 目标检测结果的工具函数和类。
"""
from ultralytics.engine.results import Results, Boxes
from numpy import ndarray
from torch import Tensor
from ultralytics.utils.ops import xywh2xyxy

class DetectedObject:
    """
    检测到的目标对象类。
    
    封装了检测到的单个目标的相关信息，包括类别名称、置信度和边界框坐标。
    
    Attributes:
        class_name (str): 目标的类别名称。
        confidence (float): 检测置信度，范围为 0 到 1。
        box (Tensor | ndarray): 目标的边界框坐标，格式为 [x1, y1, x2, y2]。
    """
    def __init__(self, class_name, confidence, box):
        """
        初始化检测到的目标对象。
        
        目标对象类封装了检测到的单个目标的相关信息，包括类别名称、置信度和边界框坐标。
        
        Args:
            class_name (str): 目标的类别名称。
            confidence (float): 检测置信度。
            box (Tensor | ndarray): 目标的边界框坐标，格式为 [x1, y1, x2, y2]。
        """
        self.class_name = class_name
        self.confidence = confidence
        self.box: Tensor | ndarray = box

    def __repr__(self):
        """返回对象的字符串表示。"""
        return f"{self.class_name} {self.confidence:.2f} {self.box}\n"
    

def ultralytics_results_to_detected_objects(results: Results) -> list[DetectedObject]:
    """
    将 Ultralytics 结果转换为 DetectedObject 对象列表。
    
    Args:
        results (Results): YOLO 模型的检测结果。
    
    Returns:
        list[DetectedObject]: 检测到的目标对象列表。
    """
    boxes: Boxes = results.boxes
    names: list[str] = results.names
    return [
        DetectedObject(names[class_id], confidence, box)
        for class_id, confidence, box
        in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy)
    ]
    
def is_results_empty(results: Results) -> bool:
    """检查结果是否为空（即没有检测到任何目标）。"""
    return len(results.boxes) == 0

def count_detected_objects(detected_objs: list[DetectedObject]) -> dict[str, int]:
    """
    统计检测到的目标对象的类别数量。
    
    Args:
        detected_objs (list[DetectedObject]): 检测到的目标对象列表。
        
    Returns:
        dict[str, int]: 每个类别的目标数量字典，键为类别名称，值为目标数量。
    """
    class_counts = {}
    for obj in detected_objs:
        class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
    return class_counts
    
def xyxy_to_xywh(xyxy: Tensor | ndarray) -> Tensor | ndarray:
    """
    将边界框的 xyxy 格式转换为 xywh 格式。
    
    Args:
        xyxy (Tensor | ndarray): 边界框坐标，格式为 [x1, y1, x2, y2]。
    
    Returns:
        Tensor | ndarray: 转换后的边界框坐标，格式为 [x, y, width, height]。
    """
    return xywh2xyxy(xyxy)