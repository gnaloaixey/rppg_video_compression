import pandas as pd
import numpy as np
import os
import re
class DatasetLoader:
    root = ''
    loader_name = 'UBFC_Phys'
    def __init__(self,root) -> None:
        self.root = root
        pass
    def load_data(self):
        list_of_video_path = []
        list_of_ppg_data = []
        contents = os.listdir(self.root)
        for content in contents:
            content_path = os.path.join(self.root, content)  # 获取内容的完整路径
            for root, dirs, files in os.walk(content_path):
                rgb_video_path = ''
                ppg_data_path = ''
                for file in files:
                    if re.search(r'^vid_(.+)_T3\.avi$', file) :  # 找到视频文件
                        rgb_video_path = os.path.join(root, file)
                    elif re.search(r'^bvp_(.+)_T3\.csv$', file):
                        ppg_data_path = os.path.join(root, file)  # PPG文件  
                if os.path.exists(rgb_video_path) and os.path.exists(ppg_data_path):
                    list_of_video_path.append(rgb_video_path)
                    list_of_ppg_data.append( self.__ppg_reader(ppg_data_path))
        return np.array(list_of_video_path),np.array(list_of_ppg_data)
    def __ppg_reader(self,path):
        ppg_df = pd.read_csv(path,header=None)
        ppg_data = ppg_df.iloc[:,0].to_numpy(dtype=np.float64)
        return ppg_data