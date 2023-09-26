import cv2
import numpy as np
class DataGenerator:
    def generate_tensor_data(self,data):
        video_paths,ppgs = data
        b_size = 1
        for i,video_path in enumerate(video_paths):
            cap = cv2.VideoCapture(video_path)
            # Read PPG signal data
            ppg_data = ppgs[i]
            # Get video frame rate
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 获取视频总帧数
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 计算视频的总时长（单位：秒）
            total_duration = video_frames / fps
            # ppg值
            frame_time_stamps = np.linspace(0,total_duration,video_frames)
            frame_time_stamps_xp = np.linspace(0,total_duration,ppg_data.shape[0])
            interpolated_ppg = np.interp(
                frame_time_stamps,
                frame_time_stamps_xp,
                ppg_data[0]
                )
            

        # 压缩

        # 特征提取
        # 线性插值
        # 归一化
        # 返回：
        #   x:tensor,y:tensor
            pass
        return 0