import pandas as pd
import numpy as np
import os
import re
class DatasetLoader:
    root = ''
    loader_name = 'ZJXU'
    def __init__(self,root) -> None:
        self.root = root
        pass
    def load_data(self):
        list_of_preprocess_video_and_ppg = []
        contents = os.listdir(self.root)
        for content in contents:
            content_path = os.path.join(self.root, content)  # 获取内容的完整路径
            for root, dirs, files in os.walk(content_path):
                rgb_video_path = ''
                wave_data_path = ''
                for file in files:
                    if re.search(r'^usb_camera-(\d+)\.avi$', file) :  # 找到视频文件
                        rgb_video_path = os.path.join(root, file)
                    elif re.search(r'^wave-(\d+)\.csv$', file):
                        wave_data_path = os.path.join(root, file)  # PPG文件  
                if os.path.exists(rgb_video_path) and os.path.exists(wave_data_path):

                    list_of_preprocess_video_and_ppg.append((rgb_video_path, wave_data_path))
        return np.array(list_of_preprocess_video_and_ppg)
    def ppg_reader(self,path):
        ppg_df = pd.read_csv(path)
        ppg_data = ppg_df.iloc[:,1].to_numpy(dtype=np.float64)
        
        return ppg_data