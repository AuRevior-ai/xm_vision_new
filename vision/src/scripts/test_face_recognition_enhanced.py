#!/usr/bin/env python3
# 此脚本用于实时识别摄像头中的所有训练过的人脸
# 采样的人脸可以使用 take_personal_faces_enhanced.py 脚本采集，
# 接着，回到 face_identification 目录下运行 preprocess.py 脚本重新训练模型，即可分辨所有样本人脸
"""
实时人脸识别器 - 摄像头实时识别所有训练过的人脸
"""
import os
import cv2
import imutils
import pickle
import numpy as np
import time

class RealtimeFaceIdentifier:
    """
    实时人脸识别器，能够实时识别摄像头中的所有人脸
    """

    def __init__(self, base_path="/home/aurevior/test_from_linke/test_from_linke/vision/vision/src/scripts/face_identification"):
        """
        初始化实时人脸识别器
        """
        # load serialized face detector
        print("正在加载人脸检测器...")
        protoPath = os.path.join(base_path, "face_detection_model", "deploy.prototxt")
        modelPath = os.path.join(base_path, "face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel")
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # load serialized face embedding model
        print("正在加载人脸嵌入模型...")
        self.embedder = cv2.dnn.readNetFromTorch(os.path.join(base_path, "assets", "openface_nn4.small2.v1.t7"))

        # load the actual face recognition model along with the label encoder
        print("正在加载训练好的识别模型...")
        self.recognizer = pickle.loads(open(os.path.join(base_path, "output", "recognizer"), "rb").read())
        self.le = pickle.loads(open(os.path.join(base_path, "output", "le.pickle"), "rb").read())
        
        print(f"✅ 模型加载完成，可识别的人脸类别: {list(self.le.classes_)}")

    def identify_all_faces(self, frame, confidence_threshold=0.5, recognition_threshold=0.4):
        """
        识别帧中的所有人脸
        """
        # Make a copy of the frame to draw on
        annotated_frame = frame.copy()
        
        # resize the frame to have a width of 600 pixels
        frame_resized = imutils.resize(frame, width=600)
        (h, w) = frame_resized.shape[:2]
        scale_x = frame.shape[1] / w
        scale_y = frame.shape[0] / h

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame_resized, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply face detector
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        face_detections = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame_resized[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                               (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()

                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = self.le.classes_[j]

                orig_startX = int(startX * scale_x)
                orig_startY = int(startY * scale_y)
                orig_endX = int(endX * scale_x)
                orig_endY = int(endY * scale_y)

                if proba > recognition_threshold:
                    text = f"{name}: {proba * 100:.1f}%"
                    color = (0, 255, 0)  # 绿色
                else:
                    text = f"未知: {proba * 100:.1f}%"
                    name = "unknown"
                    color = (0, 0, 255)  # 红色

                cv2.rectangle(annotated_frame, (orig_startX, orig_startY), (orig_endX, orig_endY), color, 2)
                y = orig_startY - 10 if orig_startY - 10 > 10 else orig_startY + 10
                cv2.putText(annotated_frame, text, (orig_startX, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                face_detections.append({
                    'name': name,
                    'confidence': proba,
                    'bbox': (orig_startX, orig_startY, orig_endX, orig_endY),
                    'detection_confidence': confidence
                })

        return annotated_frame, face_detections

def run_realtime_face_recognition():
    """运行实时人脸识别"""
    print("🎯 实时人脸识别系统")
    print("=" * 50)
    
    # 初始化OpenCV摄像头
    print("📹 正在初始化摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✅ 摄像头初始化成功")
    
    # 初始化人脸识别器
    try:
        recognizer = RealtimeFaceIdentifier()
    except Exception as e:
        print(f"❌ 人脸识别器初始化失败: {e}")
        cap.release()
        return
    
    # 设置窗口
    window_name = 'Real-time Face Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    print("\n🚀 实时人脸识别开始...")
    print("控制说明:")
    print("- 按 'q' 退出")
    print("- 按 's' 截图保存") 
    print("- 绿色框：识别成功")
    print("- 红色框：未知人脸")
    print(f"- 可识别类别: {recognizer.le.classes_}")
    
    frame_count = 0
    recognition_stats = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            frame_count += 1
            
            try:
                annotated_frame, detections = recognizer.identify_all_faces(frame)
                
                # 统计识别结果
                for detection in detections:
                    name = detection['name']
                    recognition_stats[name] = recognition_stats.get(name, 0) + 1
                
                # 在图像上添加信息
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Faces: {len(detections)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示识别统计
                y_offset = 90
                for name, count in recognition_stats.items():
                    color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                    cv2.putText(annotated_frame, f"{name}: {count}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 25
                
                # 显示实时结果
                if detections:
                    names = [d['name'] for d in detections]
                    print(f"\r实时识别: {', '.join(names)}", end="", flush=True)
                
                cv2.imshow(window_name, annotated_frame)
                
            except Exception as e:
                print(f"\n识别过程出错: {e}")
                cv2.imshow(window_name, frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n\n用户退出系统")
                break
            elif key == ord('s'):
                filename = f"realtime_recognition_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"\n截图已保存: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n系统被中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # 打印最终统计
    print("\n📊 最终统计结果:")
    print(f"总帧数: {frame_count}")
    print("识别统计:")
    for name, count in recognition_stats.items():
        print(f"  {name}: {count} 次")

if __name__ == "__main__":
    run_realtime_face_recognition()
