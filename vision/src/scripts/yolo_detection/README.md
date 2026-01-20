# yolo_detection

YOLO detection package of vision module

这个模块提供了使用 [Ultralytics YOLO (You Only Look Once)](https://docs.ultralytics.com/) 深度学习模型进行目标检测的功能。可以检测图像中的多个物体，并返回带有标注和分类信息的结果。

封装了`YoloDetector`类，能够在实时视频流中(传入视频帧frame)使用 YOLO 进行目标检测，并借助 [Supervision](https://github.com/roboflow/supervision) 包实现了图像的标注(基本还原YOLO默认的标注)。[(代码实现的官方文档参考)](https://supervision.roboflow.com/latest/how_to/detect_and_annotate/)

封装了`DetectedObject`类，编写了更加便捷的 API 接口，可以方便地获取 YOLO 目标检测返回的数据。

---

`以下内容借助Github Copilot生成，(仅供参考)`

## 使用示例

### 基本使用

```python
import cv2
from yolo_detection.yolo_detection import YoloDetector

# 初始化检测器（使用预训练的YOLOv8模型）
detector = YoloDetector("yolov8n.pt")

# 读取图像
image = cv2.imread("example.jpg")

# 进行目标检测
annotated_frame, is_empty, detected_objects = detector.use(image)

# 显示结果
if not is_empty:
    print(f"检测到 {len(detected_objects)} 个目标:")
    for obj in detected_objects:
        print(f"- {obj.class_name}: 置信度 {obj.confidence:.2f}")
else:
    print("未检测到任何目标")

# 显示标注后的图像
cv2.imshow("Detection Result", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 使用自定义模型

```python
# 使用自定义训练的模型
detector = YoloDetector("path/to/custom_model.pt")
```

### 统计检测到的目标数量

```python
from yolo_detection.utils import count_detected_objects

# 进行检测
_, _, detected_objects = detector.use(image)

# 统计各类别目标数量
class_counts = count_detected_objects(detected_objects)
print("目标统计:")
for class_name, count in class_counts.items():
    print(f"- {class_name}: {count} 个")
```

## API 参考

### `YoloDetector` 类

主要的检测器类，用于加载模型和执行检测。

```python
detector = YoloDetector(model_path)
```

参数:
- `model`: 模型路径或预训练模型名称，如 "yolov8n.pt"

方法:
- `use(frame)`: 对输入图像执行检测，返回标注图像、空结果标志和检测到的目标列表

### `DetectedObject` 类

表示检测到的单个目标对象。

属性:
- `class_name`: 目标类别名称
- `confidence`: 检测置信度 (0-1)
- `box`: 边界框坐标，格式为 [x1, y1, x2, y2]

### 辅助函数

- `get_annotated_frame(frame, results)`: 将检测结果标注到原始图像上
- `ultralytics_results_to_detected_objects(results)`: 将检测结果转换为 `DetectedObject` 对象列表
- `is_results_empty(results)`: 检查是否没有检测到目标
- `count_detected_objects(detected_objs)`: 统计不同类别的目标数量
- `xyxy_to_xywh(xyxy)`: 转换边界框格式

## 高级使用

### 仅获取特定类别的检测结果

```python
_, _, detected_objects = detector.use(image)

# 过滤特定类别
persons = [obj for obj in detected_objects if obj.class_name == "person"]
print(f"检测到 {len(persons)} 个人")
```

### 处理视频流

```python
import cv2

cap = cv2.VideoCapture(0)  # 打开摄像头
detector = YoloDetector("yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 执行检测
    annotated_frame, _, _ = detector.use(frame)
    
    # 显示结果
    cv2.imshow("YOLO Detection", annotated_frame)
    
    if cv2.waitKey(1) == 27:  # 按ESC退出
        break

cap.release()
cv2.destroyAllWindows()
```
