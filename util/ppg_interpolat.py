import cv2
import numpy as np
def generate_interpolated_ppg(ppg_array,video_path):
    cap = cv2.VideoCapture(video_path)
    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频总帧数
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 计算视频的总时长（单位：秒）
    total_duration = video_frames / fps
    # ppg值
    frame_time_stamps = np.linspace(0,total_duration,video_frames)
    frame_time_stamps_xp = np.linspace(0,total_duration,ppg_array.shape[0])
    interpolated_ppg = np.interp(
        frame_time_stamps,
        frame_time_stamps_xp,
        ppg_array
        )
    return interpolated_ppg