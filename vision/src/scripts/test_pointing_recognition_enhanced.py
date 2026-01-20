"""增强版指向手势识别测试脚本

功能：
- 打开摄像头，实时检测手部关键点
- 当检测到“指向”手势时，在画面上用箭头显示指向方向
- 在终端持续打印：是否检测到指向手势、指向向量、指尖坐标

退出方式：按下键盘 q 键
"""

import cv2
import numpy as np

from pointing_gesture_recognition.pointing_recognition import PointingRecognizer


def draw_pointing_direction(image, index_finger_points, color=(0, 255, 0)):
    """根据食指的 3 个关键点，在图像上画出指向箭头。

    Args:
        image: BGR 图像
        index_finger_points: [tip, p1, p2]，每个是 [x, y]
        color: 线条颜色
    """
    if index_finger_points is None or len(index_finger_points) < 3:
        return image

    tip = tuple(index_finger_points[0])       # 指尖
    mid = tuple(index_finger_points[1])       # 中间关节
    base = tuple(index_finger_points[2])      # 近端关节

    # 以食指中间关节到指尖的方向作为指向方向
    dir_vec = np.array(tip) - np.array(mid)
    norm = np.linalg.norm(dir_vec)
    if norm < 1e-5:
        return image

    dir_unit = dir_vec / norm
    arrow_len = 80  # 箭头长度像素
    arrow_end = (int(tip[0] + dir_unit[0] * arrow_len),
                 int(tip[1] + dir_unit[1] * arrow_len))

    # 画箭头主线
    cv2.arrowedLine(image, tip, arrow_end, color, 3, line_type=cv2.LINE_AA, tipLength=0.25)

    # 画指尖、小圆点
    cv2.circle(image, tip, 6, (0, 0, 255), -1)
    cv2.circle(image, mid, 4, (255, 0, 0), -1)
    cv2.circle(image, base, 4, (255, 0, 0), -1)

    return image


def run_pointing_recognition_enhanced(camera_index: int = 0):
    """运行增强版指向手势识别。

    Args:
        camera_index: 摄像头索引，默认 0
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_index}")
        return

    print("📹 增强版指向手势识别已启动，按 q 退出。")

    estimator = PointingRecognizer()

    frame_count = 0
    pointing_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("未能从摄像头读取到图像，结束。")
            break

        frame_count += 1

        pointing_frame, detect_pointing, index_finger_points = estimator.use(frame)

        # 如果检测到了指向手势，就在图像上画箭头
        if index_finger_points is not None:
            pointing_count += 1
            pointing_frame = draw_pointing_direction(pointing_frame, index_finger_points)

            tip = index_finger_points[0]
            p1 = index_finger_points[1]
            # 计算指向向量（从 p1 指向 tip）
            direction_vec = np.array(tip) - np.array(p1)

            print(f"[Pointing] tip={tip}, direction_vec={direction_vec}")
        else:
            print("[No pointing gesture detected]")

        # 在左上角显示简单统计信息
        info_text = f"Frames: {frame_count}  Pointing frames: {pointing_count}"
        cv2.putText(pointing_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Pointing Recognition Enhanced", pointing_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"结束。总帧数: {frame_count}, 检测到指向手势的帧数: {pointing_count}")


if __name__ == "__main__":
    run_pointing_recognition_enhanced()
