import cv2
import dlib
import numpy as np
import math
import matplotlib.pyplot as plt
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./res/shape_predictor_68_face_landmarks.dat')

def get_face(frame,front_face = None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces == None or len(faces) <= 0:
        return None
    selected_face = None
    # 取大的人脸
    if front_face == None:
        selected_face = max(faces, key=lambda face: (face.right() - face.left()) * (face.bottom() - face.top()))
    # 取位置最接近的人脸
    else:
        f_0 = front_face
        selected_face = min(
            faces, key=lambda f:
            sum([abs(i) for i in  [f_0.left()-f.left(),f_0.right()-f.right(),f_0.bottom()-f.bottom(),f_0.top()-f.top()]])
        )
    left,top,right,bottom = selected_face.left(),selected_face.top(),selected_face.right(),selected_face.bottom()
    if left <0 or top < 0 or right < 0 or bottom < 0:
        return None
    return selected_face

def get_face_and_shape(frame,front_face = None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_frame = get_face(frame,front_face)
    # cv2.rectangle(frame,(selected_face.left(),selected_face.top()),(selected_face.right(),selected_face.bottom()),0xff0000)
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if face_frame == None:
        return None,None
    shape = predictor(gray, face_frame)
    if shape == None:
        return None,None
    return face_frame,[ {'x':int(shape.part(i).x), 'y':int(shape.part(i).y)} for i in range(shape.num_parts)]
