import cv2
import numpy as np
from singleton_pattern import load_config
from util.ppg_interpolat import generate_interpolated_ppg
from util.face_detection import read_video_and_generate_factor,get_face_shape
class DataGenerator:
    batch_size:int
    time_slice_interval:float
    time_step:float
    def __init__(self) -> None:
        config = load_config.get_config()
        data_format = config['data_format']
        self.batch_size = data_format['batch_size']
        self.time_slice_interval = data_format['time_slice_interval']
        self.time_step = data_format['time_step']
        pass
    def generate_tensor_data(self,data):
        video_paths,ppgs = data
        for i,video_path in enumerate(video_paths):
            y =  generate_interpolated_ppg(ppgs[i],video_path)
            # 压缩
            # 视频特征
            X = read_video_and_generate_factor(video_path,self.face_factor_extraction)
            # 删除压缩视频 
            X,y = self.normalization(X,y)
        
        return X,y
        #   x:tensor,y:tensor
    def face_factor_extraction(self,frame):
        return 0
    def normalization(self,X,y):
        return X,y