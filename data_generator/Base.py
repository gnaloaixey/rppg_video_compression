import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from singleton_pattern import load_config
from util.ppg_interpolat import generate_interpolated_ppg
from util.face_detection import read_video_and_generate_factor,get_face_shape
class BaseDataGenerator:
    batch_size:int
    time_slice_interval:float
    time_step:float
    def __init__(self):
        config = load_config.get_config()
        data_format = config['data_format']
        self.batch_size = data_format['batch_size']
        self.time_slice_interval = data_format['time_slice_interval']
        self.time_step = data_format['time_step']
        pass
    def print_start_reading(self):
        print(f"Start Generator Data...")
    def generate_tensor_data(self,data):
        video_paths,ppgs = data
        self.print_start_reading()
        progress_bar = tqdm(video_paths, desc="Progress",ncols=100)
        for i,video_path in enumerate(progress_bar):
            y =  generate_interpolated_ppg(ppgs[i],video_path)
            # 压缩
            # 视频特征
            X = read_video_and_generate_factor(video_path,self.face_factor_extraction)
            # 删除压缩视频

            # 规格化
            X,y = self.normalization(X,y)
        # to tensor
        return X,y
        #   x:tensor,y:tensor
    def face_factor_extraction(self,frame):
        shape = get_face_shape(frame)
        if shape == None:
            return 0
        left_eye_pt = shape.part(40)
        right_eye_pt = shape.part(43)

        left_eye_x = left_eye_pt.x
        left_eye_y = left_eye_pt.y
        
        right_eye_x = right_eye_pt.x
        right_eye_y = right_eye_pt.y

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
        sum = [0,0,0]
        count = 0
        # frame[rect_start_point[1]:rect_end_point[1],rect_start_point[0]:rect_end_point[0]]
        for x in range(rect_start_point[0],rect_end_point[0]):
            for y in range(rect_start_point[1],rect_end_point[1]):
                count += 1
                sum += frame[y,x]
                # print(sum)
        # 计算RGB平均值
        forehead_avg = sum/count
        return forehead_avg
    def normalization(self,X,y):
        sigma = 1
        b = gaussian_filter(X[:,0],sigma)
        g = gaussian_filter(X[:,1],sigma)
        r = gaussian_filter(X[:,2],sigma)
        r = (r - r.mean())/r.std()
        g = (g - g.mean())/g.std()
        b = (b - b.mean())/b.std()
        y = (y - y.mean())/y.std()
        return np.column_stack((r, g, b)),y