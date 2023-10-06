import cv2
import dlib
import numpy as np
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./res/shape_predictor_68_face_landmarks.dat')

def get_face_shape(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        return shape
def read_video_and_generate_factor(video_path,handle):
    factors = list()
    while True:
    # 打开视频文件
        video_capture = cv2.VideoCapture(video_path)
        ret, frame = video_capture.read()
        if not ret:
            break
        factors.append(handle(frame))
    return np.array(factors,dtype=np.float64)