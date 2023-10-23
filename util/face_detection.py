import cv2
import dlib
import numpy as np
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./res/shape_predictor_68_face_landmarks.dat')

def get_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces == None or len(faces) <= 0:
        return None
    return faces

def get_face_shape(frame):
    # 计算缩放比例
    target_width = 320
    scale_factor = target_width / frame.shape[1]
    target_height = int(frame.shape[0] * scale_factor)
    # 缩放图像
    resized_image = cv2.resize(frame, (target_width, target_height))

    # 使用 dlib 获取缩放后图像的关键点
    faces = get_face(resized_image)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces == None or len(faces) <= 0:
        return None
    for face in faces:
        shape = predictor(gray, face)
        return [ {'x':int(shape.part(i).x / scale_factor), 'y':int(shape.part(i).y / scale_factor)} for i in range(shape.num_parts)]
