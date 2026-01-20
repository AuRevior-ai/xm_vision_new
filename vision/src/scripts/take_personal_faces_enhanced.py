#此脚本用于打开摄像头，捕捉个人脸图像并保存，用于人脸识别模型的训练(增强版)
#支持OpenCV摄像头和Azure Kinect摄像头选择
import cv2
import numpy as np
import os
from face_identification.face_identification import FaceIdentifier
from face_identification.save_personal_faces import save_faces

def choose_camera_type():
    """选择摄像头类型"""
    print("\n🎥 请选择摄像头类型：")
    print("1. OpenCV 摄像头 (USB摄像头/笔记本内置摄像头)")
    print("   📋 特点：快速启动，兼容性好，适合快速测试")
    print("   📊 分辨率：640x480 (VGA)")
    print()
    print("2. Azure Kinect 摄像头 (高清深度摄像头)")  
    print("   📋 特点：高清画质，专业级摄像头，图像质量更好")
    print("   📊 分辨率：720P/1080P 可选")
    print("   💡 建议：用于高质量人脸采集")
    
    while True:
        choice = input("\n请输入选项 (1 或 2): ").strip()
        if choice == "1":
            return "opencv"
        elif choice == "2":
            return "kinect"
        else:
            print("❌ 无效选择，请输入 1 或 2")

def initialize_opencv_camera():
    """初始化OpenCV摄像头"""
    print("📹 正在初始化 OpenCV 摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开OpenCV摄像头")
        return None
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✅ OpenCV 摄像头初始化成功")
    return cap

def initialize_kinect_camera():
    """初始化Azure Kinect摄像头"""
    print("📹 正在初始化 Azure Kinect 摄像头...")
    try:
        # 导入Azure Kinect摄像头类
        from camera.k4acam import AzureKinectCamera
        
        # 选择预设配置
        print("\n📋 Azure Kinect 预设配置：")
        print("1. balanced - 平衡模式")
        print("   📊 分辨率：720P，深度模式：WFOV_2X2BINNED")
        print("   💡 推荐：一般用途，平衡画质和性能")
        print()
        print("2. high_quality - 高质量模式")
        print("   📊 分辨率：1080P，深度模式：NFOV_UNBINNED") 
        print("   💡 推荐：最佳画质，适合高质量人脸采集")
        print()
        print("3. fast - 快速模式")
        print("   📊 分辨率：720P，深度模式：NFOV_2X2BINNED")
        print("   💡 推荐：快速响应，适合实时应用")
        
        while True:
            preset_choice = input("请选择预设 (1-3): ").strip()
            if preset_choice == "1":
                preset = "balanced"
                break
            elif preset_choice == "2":
                preset = "high_quality"
                break
            elif preset_choice == "3":
                preset = "fast"
                break
            else:
                print("❌ 无效选择，请输入 1-3")
        
        kinect_cam = AzureKinectCamera(config_preset=preset)
        kinect_cam.start()
        print(f"✅ Azure Kinect 摄像头初始化成功 (预设: {preset})")
        return kinect_cam
    except Exception as e:
        print(f"❌ Azure Kinect 摄像头初始化失败: {e}")
        print("💡 建议：检查设备连接或使用OpenCV摄像头")
        return None

def get_frame_from_camera(camera, camera_type):
    """从摄像头获取帧"""
    if camera_type == "opencv":
        ret, frame = camera.read()
        return ret, frame
    elif camera_type == "kinect":
        try:
            # 使用Azure Kinect的get_color_frame方法
            ret, frame = camera.get_color_frame()
            return ret, frame
        except Exception as e:
            print(f"读取Kinect帧时出错: {e}")
            return False, None

def release_camera(camera, camera_type):
    """释放摄像头资源"""
    if camera_type == "opencv":
        camera.release()
    elif camera_type == "kinect":
        camera.stop()
    cv2.destroyAllWindows()

def main():
    print("人脸采集脚本 - 增强版")
    print("=" * 50)
    
    # 选择摄像头类型
    camera_type = choose_camera_type()
    
    # 初始化选中的摄像头
    if camera_type == "opencv":
        camera = initialize_opencv_camera()
    else:  # kinect
        camera = initialize_kinect_camera()
    
    if camera is None:
        print("❌ 摄像头初始化失败，程序退出")
        return
    
    # 设置窗口 - 在显示图像之前设置
    window_names = {
        'main': 'Face Collection',
        'face': 'Captured Face'
    }
    
    for window_name in window_names.values():
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
    
    try:
        estimator = FaceIdentifier()
        print("✅ 人脸识别器初始化成功")
        print(f"\n🎥 当前使用摄像头：{'OpenCV 摄像头' if camera_type == 'opencv' else 'Azure Kinect 摄像头'}")
    except Exception as e:
        print(f"❌ 人脸识别器初始化失败: {e}")
        release_camera(camera, camera_type)
        return
    
    # 等待用户输入，确保真正的交互式输入
    print("\n" + "="*50)
    while True:
        try:
            face_owner = input("请输入人脸拥有者的名称（不能为空）：").strip()
            if face_owner:
                break
            else:
                print("❌ 名称不能为空，请重新输入！")
        except (EOFError, KeyboardInterrupt):
            print("\n程序被用户取消")
            release_camera(camera, camera_type)
            return
    
    print(f"✅ 输入的人脸拥有者名称：{face_owner}")
    
    # 显示保存位置信息
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "face_identification", "dataset", face_owner)
    print(f"📁 人脸图像将保存到：{save_path}")
    print(f"💾 文件格式：000.jpg, 001.jpg, 002.jpg, ...")
    
    # 检查目录是否存在
    if os.path.exists(save_path):
        existing_files = [f for f in os.listdir(save_path) if f.endswith('.jpg')]
        if existing_files:
            print(f"⚠️  目录已存在，包含 {len(existing_files)} 个现有文件")
            overwrite = input("是否要覆盖现有数据？(y/N)：").strip().lower()
            if overwrite != 'y':
                print("用户取消操作")
                release_camera(camera, camera_type)
                return
        else:
            print("📂 目录存在但为空")
    else:
        print("📂 将创建新目录")
    
    print("\n等待3秒后开始采集...")
    import time
    for i in range(3, 0, -1):
        print(f"⏰ {i} 秒...")
        time.sleep(1)
    
    # 配置参数
    N = 100  # 需要捕捉的人脸图像数量
    STRIDE = 5  # 每隔多少帧捕捉一次人脸图像
    EXPAND_FACTOR = 0.2  # 减少扩展因子，避免边界问题
    
    # 计数器和存储
    count = 0
    taked_faces = []
    frame_count = 0
    
    print(f"\n开始采集 {face_owner} 的人脸数据...")
    print(f"目标：采集 {N} 张人脸图像")
    print("控制说明:")
    print("- 按 'q' 退出")
    print("- 按 's' 手动保存当前人脸")
    print("- 请保持人脸在摄像头前并变换角度\n")

    try:
        while True:
            ret, frame = get_frame_from_camera(camera, camera_type)
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            frame_count += 1
            
            try:
                # 直接使用人脸检测，而不是人脸识别
                # 因为在采集阶段，用户的人脸数据还不存在
                face_detected, face_region, detection_confidence = detect_face_simple(frame, estimator.detector, EXPAND_FACTOR)
                
                # 显示主窗口
                display_frame = frame.copy()
                
                # 添加信息文本
                info_text = f"Collected: {len(taked_faces)}/{N}"
                cv2.putText(display_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if face_detected:
                    status_text = f"Face detected! Confidence: {detection_confidence:.2f}"
                    color = (0, 255, 0)
                    
                    # 在原图上画出检测到的人脸框
                    if face_region is not None:
                        # 简单地在原图上画个绿框表示检测到人脸
                        h, w = display_frame.shape[:2]
                        cv2.rectangle(display_frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 3)
                else:
                    status_text = "No face detected"
                    color = (0, 0, 255)
                
                cv2.putText(display_frame, status_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 添加帧计数
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(window_names['main'], display_frame)
                
                # 如果检测到人脸，显示并保存
                if face_detected and face_region is not None:
                    cv2.imshow(window_names['face'], face_region)
                    
                    # 按间隔保存人脸
                    if count % STRIDE == 0:
                        taked_faces.append(face_region)
                        print(f"采集进度: {len(taked_faces)}/{N} ({len(taked_faces)/N*100:.1f}%)")
                    
                    count += 1
                
                # 检查是否完成采集
                if len(taked_faces) >= N:
                    print(f"\n✅ 采集完成！共采集 {len(taked_faces)} 张人脸图像")
                    break
                
            except Exception as e:
                print(f"处理帧时出错: {e}")
                # 即使出错也显示原始帧
                cv2.imshow(window_names['main'], frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n用户退出，已采集 {len(taked_faces)} 张图像")
                break
            elif key == ord('s') and face_detected:
                # 手动保存当前人脸
                if face_region is not None:
                    taked_faces.append(face_region)
                    print(f"手动保存: {len(taked_faces)}/{N}")

    except KeyboardInterrupt:
        print(f"\n程序被中断，已采集 {len(taked_faces)} 张图像")
    
    finally:
        release_camera(camera, camera_type)
    
    # 保存采集的人脸
    if len(taked_faces) > 0:
        print(f"\n💾 正在保存 {len(taked_faces)} 张人脸图像...")
        
        try:
            # 直接在这里实现保存逻辑，避免外部函数问题
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "face_identification", "dataset", face_owner)
            
            # 创建目录
            os.makedirs(save_dir, exist_ok=True)
            
            # 清空目录（如果有旧文件）
            for file in os.listdir(save_dir):
                if file.endswith('.jpg'):
                    os.remove(os.path.join(save_dir, file))
            
            # 保存新文件
            saved_count = 0
            for i, face_img in enumerate(taked_faces):
                filename = os.path.join(save_dir, f"{str(i).zfill(3)}.jpg")
                if cv2.imwrite(filename, face_img):
                    saved_count += 1
                else:
                    print(f"⚠️ 保存失败: {filename}")
            
            # 显示详细的保存信息
            print(f"✅ 人脸图像保存成功！")
            print(f"📁 保存位置：{save_dir}")
            print(f"📋 保存详情：")
            print(f"   - 总文件数：{saved_count} 张")
            print(f"   - 文件名格式：000.jpg ~ {str(len(taked_faces)-1).zfill(3)}.jpg")
            print(f"   - 文件大小：约 {saved_count * 20} KB") 
            
            # 验证文件是否真的保存了
            if os.path.exists(save_dir):
                saved_files = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
                print(f"✓ 验证：实际保存了 {len(saved_files)} 个文件")
                
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            print("💡 建议：检查磁盘空间和文件权限")
    else:
        print("⚠️ 没有采集到任何人脸图像，无需保存")
    
    print("\n🎉 程序结束")

def detect_face_simple(frame, detector, expand_factor):
    """
    简单的人脸检测函数，返回是否检测到人脸、人脸区域和置信度
    """
    try:
        # 调整图像大小
        (h, w) = frame.shape[:2]
        
        # 构建blob进行人脸检测
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        
        # 进行人脸检测
        detector.setInput(imageBlob)
        detections = detector.forward()
        
        # 寻找最佳检测结果
        best_detection = None
        best_confidence = 0
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_confidence:
                best_confidence = confidence
                best_detection = detections[0, 0, i, 3:7]
        
        if best_detection is not None:
            # 计算边界框坐标
            box = best_detection * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # 扩展边界框
            fH, fW = endY - startY, endX - startX
            startX = int(max(0, startX - fW * expand_factor))
            endX = int(min(w, endX + fW * expand_factor))
            startY = int(max(0, startY - fH * expand_factor))
            endY = int(min(h, endY + fH * expand_factor))
            
            # 提取人脸区域
            face = frame[startY:endY, startX:endX]
            
            # 确保人脸区域有效
            if face.shape[0] > 20 and face.shape[1] > 20:
                return True, face, best_confidence
        
        return False, None, 0.0
    
    except Exception as e:
        print(f"人脸检测时出错: {e}")
        return False, None, 0.0

if __name__ == "__main__":
    main()
