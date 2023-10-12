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
    video_capture = cv2.VideoCapture(video_path)
    video_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    temp = None
    while True:
    # 打开视频文件
        ret, frame = video_capture.read()
        if not ret:
            break
        x = handle(frame)
        if x != None:
            temp = x
            factors.append(x)
        elif x == None and temp != None:
            factors.append(temp)
        
    return np.array(factors,dtype=np.float64)