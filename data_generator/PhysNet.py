import cv2
from common.cache import CacheType
from data_generator.Base import BaseDataGenerator as Base
import numpy as np
class DataGenerator(Base):
    def __init__(self, cache_type=CacheType.TRAIN):
        super().__init__(cache_type)
        self.frame = None
    def __normalization__(self,X,y):
        # C,T,W,H
        X = X.transpose((3, 0, 2, 1))
        X = X/255
        y = (y - y.min())/(y.max() -y.min())
        y = (y-y.mean())/y.std()
        return X,y
    def __face_factor_extraction__(self,frame,face,face_shapes):
        height, width, _ = frame.shape
    


        if self.frame is not None:
            gra_1 = frame[:,:,1]
            gra_2 = self.frame[:,:,1]
            f = np.abs(gra_1 - gra_2)
            f = 5 * np.power(f,2)
            f = cv2.GaussianBlur(f,(5,5),0,0)
            new_frame = np.zeros_like(frame)
            new_frame[:,:,1] = f
            for face_shape in face_shapes:
                cv2.rectangle(new_frame,(face.left(),face.top()),(face.right(),face.bottom()),0xff0000)
                cv2.circle(new_frame, (face_shape['x'], face_shape['y']), 2, (0, 0, 255),-1)
            cv2.imshow("Frame",new_frame)
            cv2.waitKey(1)
        self.frame = frame

        # 获取人脸区域的左上角和右下角坐标
        top_left = (face.left(), face.top())
        bottom_right = (face.right(), face.bottom())
        frame_face = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),0xff0000)
        # cv2.imshow('Video', frame_face)
        # cv2.waitKey(10)

        target_height = 180
        target_width = 180
        frame_face = cv2.resize(frame_face, (target_width, target_height),interpolation=cv2.INTER_CUBIC)

        return frame_face
