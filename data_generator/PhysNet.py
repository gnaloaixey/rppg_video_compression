import cv2
from data_generator.Base import BaseDataGenerator as Base
import numpy as np
class DataGenerator(Base):
    def __normalization__(self,X,y):
        # C,T,W,H
        X = X.transpose((3, 0, 2, 1))
        X = X/255
        y = (y - y.min())/(y.max() -y.min())
        return X,y
    def __face_factor_extraction__(self,frame,face,shape):
        height, width, _ = frame.shape

        # 获取人脸区域的左上角和右下角坐标
        top_left = (face.left(), face.top())
        bottom_right = (face.right(), face.bottom())
        frame_face = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),0xff0000)
        # cv2.imshow('Video', frame_face)
        # cv2.waitKey(10)

        target_height = 128
        target_width = 128
        frame_face = cv2.resize(frame_face, (target_width, target_height),interpolation=cv2.INTER_CUBIC)

        return frame_face
