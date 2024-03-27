# %%
import ffmpeg
# config
import common.init as init
from singleton_pattern.load_model import get_model
from  singleton_pattern.load_config import get_config
from singleton_pattern.load_dataset_reader import get_test_reader
from singleton_pattern.load_data_generator import create_tensor_data_generator
from common.cache import Cache
from common.cuda_info import get_device
from singleton_pattern.load_config import get_non_dnn_method_list
from common.cache import CacheType
from singleton_pattern.load_model import get_model
from loss.pearson import PearsonLoss
import shutil
import os
import numpy as np
import cv2
import concurrent.futures  
import traceback
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
# config_name = 'PhysNet.yaml'
config_name = 'POS.yaml'
# config_name = 'PCA.yaml'


# %%



init.run(config_name,False)

root_output_path = './cache/COMPRESS'
os.makedirs(root_output_path, exist_ok=True)
config = get_config()
method = config.get('method')
config['test']['dataset']['force_clear_cache'] = True
ploss = PearsonLoss()
# 创建互斥锁
save_psnr_to_sheet_lock = threading.Lock()
save_pearson_and_snr_lock = threading.Lock()

def calculate(codec,suffix, crf = None, qp = None):
    mode = 'crf' if qp == None else 'qp'
    intensity = crf if qp == None else qp
    # load test dataset
    test_reader = get_test_reader()
    videos,ppgs = test_reader.load_data(print_info = False)
    codec_videos = []
    for i,video_path in enumerate(videos):
        if intensity == 0:
            codec_videos.append(video_path)
            continue
        output_video_root = f'{root_output_path}/{codec}'
        os.makedirs(output_video_root, exist_ok=True)
        output_video_path = f'{output_video_root}/{mode}={str(intensity)}_{str(i)}.{suffix}'
        codec_videos.append(os.path.abspath(output_video_path))

        compress(codec,video_path,output_video_path,mode,intensity)
    '''
        video psnr and ssim
    '''
    videos_psnrs = []
    for index in range(len(videos)):
        psnrs = []
        cap1 = cv2.VideoCapture(videos[index])  
        cap2 = cv2.VideoCapture(codec_videos[index])
        f_i = 0
        while True:
            ret1, frame_1 = cap1.read()
            ret2, frame_2 = cap2.read()
            if not ret1 or not ret2:
                print(codec_videos[index])
                print('read video error:',codec)
                cap1.release()
                cap2.release()
                break
            subtracted_frame = cv2.absdiff(frame_1, frame_2)
            gray_frame = cv2.cvtColor(subtracted_frame,cv2.COLOR_BGR2GRAY)
            gamma = 0.3
            gamma_corrected_frame = np.power(gray_frame / 255.0, gamma) * 255.0
            gamma_corrected_frame = np.uint8(gamma_corrected_frame)
            if f_i > 10:
                break
            if f_i > 200 and f_i < 220:
                diff_root = f'out/img/diff/{codec}'
                os.makedirs(diff_root, exist_ok=True)
                cv2.imwrite(f'{diff_root}/{mode}={str(intensity)}_{str(f_i)}.png',cv2.cvtColor(gamma_corrected_frame,cv2.COLOR_GRAY2BGR))
            psnr = PSNR(frame_1,frame_2)
            psnrs.append(psnr)
            f_i += 1
        videos_psnrs.append(psnrs)
        cap1.release()
        cap2.release()
    videos_psnrs = np.array(videos_psnrs).T
    # 保存
    with save_psnr_to_sheet_lock:
        sheet_name = f'{codec}_{mode}_{str(intensity)}'
        save_psnr_to_sheet(sheet_name,videos_psnrs)
    '''
        pearson and snr
    '''
    test_data_generator = create_tensor_data_generator(CacheType.TEST_SYNC)
    test_dataloader = test_data_generator.get_tensor_dataloader((codec_videos,ppgs),print_info = False)
    non_dnn_method_list = get_non_dnn_method_list()
    is_need_train = config['method'] not in non_dnn_method_list
    model = get_model()
    if is_need_train:
        cache_model = Cache(CacheType.MODEL).read_model()
        model.load_state_dict(cache_model.state_dict())
    model.eval()
    gpu_device = get_device()
    model.to(gpu_device)

    pearsons = []
    snrs = []
    for batch_X, batch_y in test_dataloader:
        batch_X = batch_X.to(gpu_device)
        batch_y = batch_y.cpu()
        outputs = model.forward(batch_X).cpu()
        p = ploss.forward(batch_y,outputs)
        s = SNR(y_1,y_2)
        pearsons.append(p)
        snrs.append(s)
        print(p,s)
        # 画图
        y_1 = batch_y.numpy().flatten()
        y_2 = outputs.numpy().flatten()
        # print(y_1,y_2)
        plt.plot(y_1)
        plt.plot(y_2)
        plt.show()
    datas = {
        'pearson':np.array(pearsons),
        'snr':np.array(snrs)
    }
    with save_pearson_and_snr_lock:
        sheet_name = f'{codec}_{mode}_{str(intensity)}'
        save_pearson_and_snr(sheet_name,datas)


def compress(codec,video_path,output_video_path,mode,intensity = None,):
    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    command = ['ffmpeg', '-i', video_path, '-c:v', codec, '-crf', str(intensity),'-preset','medium','-pix_fmt','yuv422p','-gpu','auto', output_video_path]
    subprocess.run(command)

    # param = {
    #     'codec':codec,
    #     'pix_fmt':'yuv420p'
    # }
    # param[mode] = intensity

    # steam = ffmpeg.input(video_path)
    # steam = steam.output(output_video_path, **param,)
    # steam.run()
    # print(steam)


def save_psnr_to_sheet(sheet_name,datas):
    file_path = 'out/psnr.xlsx'
    if not os.path.exists(file_path):
        pd.DataFrame().to_excel(file_path)
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        try:
            writer.book.remove(writer.book[sheet_name])
        except:
            pass
        df = pd.DataFrame(datas)
        df.to_excel(writer, index=False, sheet_name=sheet_name,header=[f'video_{str(i+1)}' for i in range(datas.shape[1])])

def save_pearson_and_snr(sheet_name,datas):
    file_path = 'out/pearson_and_snr.xlsx'
    if not os.path.exists(file_path):
        pd.DataFrame().to_excel(file_path)
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        try:
            writer.book.remove(writer.book[sheet_name])
        except:
            pass
        df = pd.DataFrame(datas)
        df.to_excel(writer, index=False, sheet_name=sheet_name,header=True)

def PSNR(frame_1,frame_2):
    f_pow = np.power(frame_1 - frame_2,2)
    average_per_channel = np.mean(f_pow, axis=-1)
    mse = average_per_channel/(frame_1.shape[0] * frame_1.shape[1])
    if (np.sum(mse)/3) == 0:
        return np.inf
    return 10 * np.log10(np.power(255,2) / (np.sum(mse)/3))

def SNR(y_1,y_2):
    y_1 = y_1.cpu().numpy().flatten()
    y_2 = y_2.cpu().numpy().flatten()
    # 计算信号和噪声的功率
    signal_power = np.mean(y_1 ** 2)
    noise_power = np.mean((y_1 - y_2) ** 2)
    # 检查噪声功率是否为0
    if noise_power == 0:
        return float('inf')  # 返回正无穷，表示信号远远大于噪声
    # 计算信噪比
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

def tesk_wrap(*args, **kwargs):
    calculate(*args, **kwargs)
    try:
        pass
    except Exception  as e:
        print(e)
        traceback.print_exc()
    print(*args, str(kwargs))
    # print(str(threading.get_ident()))



# %%
codes = [('h264','avi'),('hevc','mp4'),('av1','webm'),('mpeg4','avi')]
codes = [('hevc','mp4')]
codes = [('h264','avi')]
codes = [('libaom-av1','mkv')]
for codec,suffix in codes:
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     tasks = [executor.submit(tesk_wrap, codec,crf=crf) for crf in range(20, 30)]
    #     for future in concurrent.futures.as_completed(tasks):
    #         pass
    for crf in range(30,31):
        tesk_wrap(codec,suffix,crf=crf)
        break

# %%
# shutil.rmtree(root_output_path,ignore_errors=True)


