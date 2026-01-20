# 增强版手势识别测试脚本
# 通过测试，可以在xiaomeng_env环境下使用
import cv2
from gesture_recognition.gesture_recognition import GestureRecognizer
import time

def main():
    print("手势识别测试脚本 - 增强版")
    print("=" * 50)
    print("支持的手势类型:")
    print("- Pointing_Up (手指向上)")  
    print("- Victory (胜利手势/剪刀手)")
    print("- Open_Palm (张开手掌)")
    print("- Closed_Fist (握拳)")
    print("- Thumb_Up (拇指向上)")
    print("- 等等...")
    print("\n控制:")
    print("- 按 'q' 退出")
    print("- 按 's' 截图保存")
    print("- 按 'r' 重置统计")
    print("\n开始手势识别...")

    # 初始化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    estimator = GestureRecognizer()
    
    # 设置窗口属性 - 只需要设置一次
    window_name = 'Gesture Recognition'  # 使用英文名称
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 允许调整窗口大小
    cv2.resizeWindow(window_name, 960, 720)  # 设置更大的窗口大小
    
    # 统计信息
    frame_count = 0
    gesture_counts = {}
    start_time = time.time()
    last_gesture = None
    last_confidence = 0
    screenshot_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break

            frame_count += 1
            
            # 手势识别
            try:
                annotated_frame, has_gesture, category_name, score = estimator.use(frame)
                
                # 统计手势
                if category_name and category_name != "None":
                    if category_name not in gesture_counts:
                        gesture_counts[category_name] = 0
                    gesture_counts[category_name] += 1
                    last_gesture = category_name
                    last_confidence = score if score else 0

                # 在图像上添加简洁的英文信息
                if annotated_frame is not None:
                    display_frame = annotated_frame.copy()
                else:
                    display_frame = frame.copy()

                # 只显示当前检测到的手势（英文）
                if category_name and category_name != "None" and score:
                    text = f"Gesture: {category_name} ({score:.2f})"
                    color = (0, 255, 0)  # 绿色
                    cv2.putText(display_frame, text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 显示FPS信息（英文）
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                           (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 显示图像
                # 确保图像数据正确并显示
                if display_frame is not None and display_frame.shape[0] > 0 and display_frame.shape[1] > 0:
                    cv2.imshow(window_name, display_frame)
                else:
                    cv2.imshow(window_name, frame)

                # 控制台输出（每30帧输出一次详细信息）
                if frame_count % 30 == 0:
                    print(f"\n帧 {frame_count}: FPS={fps:.1f}")
                    if category_name and score:
                        print(f"  当前手势: {category_name} (置信度: {score:.3f})")
                    
                    if gesture_counts:
                        print("  手势统计:")
                        for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
                            percentage = count / frame_count * 100
                            print(f"    {gesture}: {count} 次 ({percentage:.1f}%)")

            except Exception as e:
                print(f"处理帧时出错: {e}")
                cv2.imshow(window_name, frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户退出程序")
                break
            elif key == ord('s'):
                # 保存截图
                screenshot_count += 1
                filename = f"gesture_screenshot_{screenshot_count}.jpg"
                if 'display_frame' in locals():
                    cv2.imwrite(filename, display_frame)
                    print(f"已保存截图: {filename}")
                else:
                    cv2.imwrite(filename, frame)
                    print(f"已保存截图: {filename}")
            elif key == ord('r'):
                # 重置统计
                gesture_counts.clear()
                frame_count = 0
                start_time = time.time()
                print("统计信息已重置")

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")

    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 最终统计
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print("手势识别会话总结")
        print('='*50)
        print(f"总运行时间: {total_time:.1f} 秒")
        print(f"总处理帧数: {frame_count}")
        print(f"平均FPS: {frame_count/total_time:.1f}")
        
        if gesture_counts:
            print(f"\n检测到的手势:")
            for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / frame_count * 100
                print(f"  {gesture}: {count} 次 ({percentage:.1f}%)")
        else:
            print("未检测到任何手势")
            
        print(f"\n保存的截图: {screenshot_count} 张")
        print("程序结束")

if __name__ == "__main__":
    main()
