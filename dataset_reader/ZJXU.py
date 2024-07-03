import pandas as pd
import numpy as np
import os
from common.import_tqdm import tqdm
import re
from dataset_reader.Base import BaseDatasetReader
class DatasetReader(BaseDatasetReader):
    root = ''
    loader_name = 'ZJXU'
    def load_data(self,print_info = True):
        list_of_video_path = []
        list_of_ppg_data = []
        contents = os.listdir(self.root)
        if print_info:
            self.print_start_reading()
        progress_bar = tqdm(contents, desc="Progress") if print_info else contents
        for content in progress_bar:
            content_path = os.path.join(self.root, content)  # 获取内容的完整路径
            for root, dirs, files in os.walk(content_path):
                rgb_video_path = ''
                ppg_data_path = ''
                for file in files:
                    if re.search(r'^usb_camera-(\d+)\.avi$', file) :  # 找到视频文件
                        rgb_video_path = os.path.join(root, file)
                    elif re.search(r'^wave-(\d+)\.csv$', file):
                        ppg_data_path = os.path.join(root, file)  # PPG文件  
                if os.path.exists(rgb_video_path) and os.path.exists(ppg_data_path):
                    list_of_video_path.append(rgb_video_path)
                    list_of_ppg_data.append( self.__ppg_reader(ppg_data_path))
        if print_info:
            progress_bar.clear()
            progress_bar.close()
        return np.array(list_of_video_path),np.array(list_of_ppg_data)
    def __ppg_reader(self,path):
        ppg_df = pd.read_csv(path)
        ppg_data = ppg_df.iloc[:,1].to_numpy(dtype=np.float64)
        
        return ppg_data