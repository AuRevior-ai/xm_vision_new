import cv2
import numpy as np
from face_identification.face import FaceIdentifier
from face_identification.save_personal_faces import save_faces
import pyk4a
from pyk4a import PyK4A, ColorResolution, Config
class RememberFace:
    def __init__(self, num_faces=50, stride=5, expand_factor=0.5):
        self.k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
                synchronized_images_only=True,
            )
        )
        self.estimator = FaceIdentifier()
        self.N = num_faces
        self.STRIDE = stride
        self.EXPAND_FACTOR = expand_factor
        self.count = 0
        self.taked_faces = []

    def start_camera(self):
        self.k4a.start()

    def stop_camera(self):
        self.k4a.stop()
        cv2.destroyAllWindows()

    def capture_faces(self, frame, name):
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        face_identified_frame, faces_amount, faces_boxes = self.estimator.use(frame)

        # cv2.imshow('frame', face_identified_frame)

        if faces_amount > 0:
            (h, w) = frame.shape[:2]
            box = faces_boxes[0] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            fH, fW = endY - startY, endX - startX
            startX = int(max(0, startX - fW * self.EXPAND_FACTOR))
            endX = int(min(w, endX + fW * self.EXPAND_FACTOR))
            startY = int(max(0, startY - fH * self.EXPAND_FACTOR))
            endY = int(min(h, endY + fH * self.EXPAND_FACTOR))
            face = frame[startY:endY, startX:endX]
            # print(startX, startY, endX, endY)
            # cv2.imshow('taked_face', face)

            if self.count % self.STRIDE == 0:
                self.taked_faces.append(face)
            self.count += 1

        if len(self.taked_faces) >= self.N:
            save_faces(name, self.taked_faces)
            # self.stop_camera()
            return 'succeed'
        else:
            return 'no_finish'
'''
if __name__ == "__main__":

    config = Config(color_resolution=ColorResolution.RES_1080P)
    camera = PyK4A(config)
    camera.start()
    face_capture = RememberFace()
    while True:
        capture = camera.get_capture()
        color_image = capture.color
        a = face_capture.capture_faces(color_image, '7')
        print(a)
'''