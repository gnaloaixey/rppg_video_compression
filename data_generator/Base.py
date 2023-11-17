import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import gaussian_filter
from util.import_tqdm import tqdm
from singleton_pattern import load_config
from util.ppg_interpolat import generate_interpolated_ppg_by_video_capture
from util.face_detection import get_face_shape
from util.torch_info import get_device_str
from util.cache import Cache,CacheDataset

class BaseDataGenerator:
    def __init__(self,dataset_type='train'):
        self.dataset_type = dataset_type
        config = load_config.get_config()
        data_format = config['data_format']
        self.batch_size = data_format['batch_size']
        self.slice_interval = data_format['slice_interval']
        self.step = data_format['step']
        pass
    def print_start_reading(self):
        print(f"Start Generator Data...")
    def get_tensor_dataloader(self,data:[np.array,np.array] or None,force_clear_cache = False,shuffle = False,num_workers=8,pin_memory=True,):
        cache = Cache(self.dataset_type)
        if force_clear_cache:
            cache.free()
        if not cache.exist() or cache.size() == 0:
            self.__generate_cache__(data,cache)
        dataset = CacheDataset(cache)
        print(f'dataset size: {len(dataset)}')
        data_loader = DataLoader(dataset, batch_size=self.batch_size,
                                 num_workers=num_workers,pin_memory=pin_memory,
                                 pin_memory_device=get_device_str(), shuffle=shuffle)
        return data_loader
    def __generate_cache__(self,data:[np.array,np.array],cache:Cache):
        video_paths,ppgs = data
        self.print_start_reading()
        # Get video frame rate
        config = load_config.get_config()
        fps = config['data_format']['fps']
        step = int(self.step)
        slice_interval = int(self.slice_interval)

        dataset_index = 0
        progress_bar = tqdm(video_paths, desc="Progress")
        for i,video_path in enumerate(progress_bar):
            # compress
            compressed_path = video_path
            pass
            # open video
            video_capture = cv2.VideoCapture(compressed_path)
            interpolated_ppg =  generate_interpolated_ppg_by_video_capture(ppgs[i],video_capture)
            # Set the frame rate of the video capture object
            video_capture.set(cv2.CAP_PROP_FPS, fps)

            y_queue = list()
            factor_queue = list()
            progress_video_bar = tqdm(interpolated_ppg, desc=f"Processing videos {i+1}{' 'if i <10 else ''}")
            for ppg_strength in progress_video_bar:
                ret, frame = video_capture.read()
                if not ret:
                    continue
                shape = get_face_shape(frame)
                if shape == None:
                    factor_queue.clear()
                    y_queue.clear()
                    continue
                x = self.__face_factor_extraction__(frame,shape)
                factor_queue.append(x)
                y_queue.append(ppg_strength)
                if len(y_queue) >= slice_interval:
                    y_temp = np.array(y_queue)
                    factor_temp = np.array(factor_queue)
                    factor_temp,y_temp = self.__normalization__(factor_temp,y_temp)
                    cache.save(factor_temp,y_temp,dataset_index)
                    dataset_index += 1
                    y_queue = y_queue[step:]
                    factor_queue = factor_queue[step:]
            progress_video_bar.clear()
            progress_video_bar.close()
            # 删除压缩视频
            pass
    def __face_factor_extraction__(self,frame:np.array,shape:list or None):
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
    def __normalization__(self,X:np.array,y:np.array):
        sigma = 1
        b = gaussian_filter(X[:,0],sigma)
        g = gaussian_filter(X[:,1],sigma)
        r = gaussian_filter(X[:,2],sigma)
        r = (r - r.mean())/r.std()
        g = (g - g.mean())/g.std()
        b = (b - b.mean())/b.std()
        y = (y - y.mean())/y.std()
        return np.column_stack((r, g, b)),y

