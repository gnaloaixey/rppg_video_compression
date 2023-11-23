import cv2
from data_generator.Base import BaseDataGenerator as Base
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from common.cuda_info import get_device
class DataGenerator(Base):
    def __normalization__(self,X,y):
        sigma = 1
        g = gaussian_filter(X,sigma)
        g = (g - g.mean())/g.std()
        y = (y - y.mean())/y.std()
        return g,y
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


        rect_frame = frame[rect_start_point[1]:rect_end_point[1],rect_start_point[0]:rect_end_point[0],1]
        X_tensor = torch.tensor(rect_frame, dtype=torch.float64).to(get_device())

        # 计算g通道的平均值
        mean_g = torch.mean(X_tensor)

        return mean_g.to('cpu').item()
        cv2.imshow('Video', frame)
        cv2.waitKey(1)