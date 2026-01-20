#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
#此文件用于可视化手势识别的结果
from matplotlib import pyplot as plt
import mediapipe as mp

# 兼容新版本 mediapipe 的导入方式
try:
    # 尝试新版本的导入路径
    from mediapipe.tasks.python.components.containers import landmark as landmark_module
    # 创建一个兼容的 landmark_pb2 替代
    class LandmarkCompat:
        def __init__(self):
            pass
        
        class NormalizedLandmarkList:
            def __init__(self):
                self.landmark = []
        
        class NormalizedLandmark:
            def __init__(self, x=0, y=0, z=0):
                self.x = x
                self.y = y
                self.z = z
    
    landmark_pb2 = LandmarkCompat()
    landmark_pb2.NormalizedLandmark = LandmarkCompat.NormalizedLandmark
    landmark_pb2.NormalizedLandmarkList = LandmarkCompat.NormalizedLandmarkList
    print("使用新版本 mediapipe 导入路径")
except ImportError:
    # 尝试旧版本的导入路径
    try:
        from mediapipe.framework.formats import landmark_pb2
        print("使用旧版本 mediapipe 导入路径")
    except ImportError:
        print("警告: 无法导入 landmark 模块，某些可视化功能可能不可用")
        # 创建一个虚拟的 landmark_pb2
        class DummyLandmark:
            def __init__(self, x=0, y=0, z=0):
                self.x = x
                self.y = y
                self.z = z
                
        class DummyLandmarkList:
            def __init__(self):
                self.landmark = []
                
        class DummyLandmarkPb2:
            NormalizedLandmark = DummyLandmark
            NormalizedLandmarkList = DummyLandmarkList
        landmark_pb2 = DummyLandmarkPb2()

import math
import cv2

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

# 兼容新版本 mediapipe - 处理 solutions 模块
try:
    # 尝试旧版本的 solutions API
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    print("使用旧版本 mediapipe solutions")
except AttributeError:
    # 新版本没有 solutions，创建兼容层
    print("新版本 mediapipe - 创建兼容层")
    
    class HandsCompat:
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (9, 10), (10, 11), (11, 12),     # 中指
            (13, 14), (14, 15), (15, 16),    # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            (5, 9), (9, 13), (13, 17)        # 手掌连接
        ]
    
    class DrawingCompat:
        @staticmethod
        def draw_landmarks(image, landmarks, connections=None, landmark_drawing_spec=None, connection_drawing_spec=None):
            # 简化的绘制函数
            if landmarks and hasattr(landmarks, 'landmark'):
                # 绘制关键点
                for landmark in landmarks.landmark:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                
                # 绘制连接线（如果提供了连接信息）
                if connections:
                    for connection in connections:
                        try:
                            start_idx, end_idx = connection
                            if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
                                start_landmark = landmarks.landmark[start_idx]
                                end_landmark = landmarks.landmark[end_idx]
                                
                                start_point = (int(start_landmark.x * image.shape[1]),
                                             int(start_landmark.y * image.shape[0]))
                                end_point = (int(end_landmark.x * image.shape[1]),
                                           int(end_landmark.y * image.shape[0]))
                                
                                cv2.line(image, start_point, end_point, (255, 255, 255), 2)
                        except:
                            continue  # 跳过无效的连接
    
    class DrawingStylesCompat:
        @staticmethod
        def get_default_hand_landmarks_style():
            return {}
        
        @staticmethod
        def get_default_hand_connections_style():
            return {}
    
    mp_hands = HandsCompat()
    mp_drawing = DrawingCompat()
    mp_drawing_styles = DrawingStylesCompat()


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


def draw_one_image_with_gestures_and_hand_landmarks(image, gesture, multi_hand_landmarks):
    """Draws the gesture category and its score along with the hand landmarks on the image."""
    title = f"{gesture.category_name} ({gesture.score:.2f})"
    
    # 获取图像数据 - 确保正确的格式
    if hasattr(image, 'numpy_view'):
        # 如果是 MediaPipe Image 对象
        annotated_image = image.numpy_view().copy()
        
        # 检查图像是RGB还是BGR，MediaPipe通常是RGB格式
        if annotated_image.shape[2] == 3:  # 确保是3通道
            # convert the image from RGB to BGR for OpenCV
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    else:
        # 如果已经是numpy数组
        annotated_image = image.copy()
    
    # 绘制手部关键点
    for hand_landmarks in multi_hand_landmarks:
        try:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        except Exception as e:
            # 简单绘制关键点作为备选方案
            for landmark in hand_landmarks:
                x = int(landmark.x * annotated_image.shape[1])
                y = int(landmark.y * annotated_image.shape[0])
                cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
        
    # put the gesture category and its score on the image
    cv2.putText(annotated_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated_image