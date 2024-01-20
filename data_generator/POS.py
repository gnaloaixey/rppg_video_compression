import cv2
from data_generator.Base import BaseDataGenerator as Base
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from common.cuda_info import get_device
class DataGenerator(Base):
    def __normalization__(self,X,y):
        # (T, C, H,W)
        X = X.transpose((0, 3, 1,2))
        X = X/255
        y = (y - y.min())/(y.max() -y.min())
        return X,y
    def __face_factor_extraction__(self,frame,shape):
        left_eye_pt = shape[40]
        right_eye_pt = shape[43]
        left_eye_x = left_eye_pt['x']
        left_eye_y = left_eye_pt['y']
        
        right_eye_x = right_eye_pt['x']
        right_eye_y = right_eye_pt['y']

        temp = right_eye_x-left_eye_x
        proportional_length = 0
        proportional_length = temp
        # proportional_length = 20
        rect_center_point = np.array([(left_eye_x+right_eye_x)/2,(left_eye_y + right_eye_y)/2],dtype=np.int32)
        rect_width = proportional_length/4
        rect_height = proportional_length/8
        #向上平移一定距离
        rect_center_point[1] -= int(proportional_length/2)
        rect_start_point = rect_center_point - np.array([int(rect_width/2),rect_height/2],dtype=np.int32)
        rect_end_point = rect_center_point + np.array([int(rect_width/2),rect_height/2],dtype=np.int32)


        rect_frame = frame[rect_start_point[1]:rect_end_point[1],rect_start_point[0]:rect_end_point[0],:]

        return_frame = np.resize(rect_frame,(128,128,3))
        # cv2.rectangle(frame,rect_start_point,rect_end_point,color=(255,255,0))
        # cv2.imshow('Video', return_frame)
        # cv2.waitKey(1)
        return return_frame