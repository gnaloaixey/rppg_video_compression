import pandas as pd
import numpy as np
from util.import_tqdm import tqdm
import os
import re

from dataset_reader.Base import DatasetReader as Base
class DatasetReader(Base):
    root = ''
    loader_name = 'UBFC_Phys'
    data_type = 1
    def __init__(self,root,data_type=1) -> None:
        self.root = root
        self.data_type = data_type
        pass
    def load_data(self):
        list_of_video_path = []
        list_of_ppg_data = []
        contents = os.listdir(self.root)
        self.print_start_reading()
        progress_bar = tqdm(contents, desc="Progress")
        for content in progress_bar:
            content_path = os.path.join(self.root, content)  # 获取内容的完整路径
            for root, dirs, files in os.walk(content_path):
                rgb_video_path = ''
                ppg_data_path = ''
                for file in files:
                    pattern_video = re.compile(r'^vid_(.+)_(T{0})\.avi$'.format(self.data_type))
                    pattern_bvp = re.compile(r'^bvp_(.+)_(T{0})\.csv$'.format(self.data_type))
                    if pattern_video.search(file) :  # 找到视频文件
                        rgb_video_path = os.path.join(root, file)
                    elif pattern_bvp.search(file):
                        ppg_data_path = os.path.join(root, file)  # PPG文件  
                if os.path.exists(rgb_video_path) and os.path.exists(ppg_data_path):
                    list_of_video_path.append(rgb_video_path)
                    list_of_ppg_data.append( self.__ppg_reader(ppg_data_path))
        progress_bar.clear()
        progress_bar.close()
        return np.array(list_of_video_path),np.array(list_of_ppg_data)
    def __ppg_reader(self,path):
        ppg_df = pd.read_csv(path,header=None)
        ppg_data = ppg_df.iloc[:,0].to_numpy(dtype=np.float64)
        return ppg_data
