import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, lfilter
import pywt

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This  is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = len(signal)

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def lowpass_filter(data, fs, cutoff_freq, order=5):
    # 设定时间向量
    t = np.arange(len(data)) / fs
    
    # 使用低通滤波器滤波
    filtered_data = butter_lowpass_filter(data, cutoff_freq, fs, order)
    
    return t, filtered_data


def wavelet_denoising(data, wavelet='db1', level=4, threshold=0.1):
    # 进行小波变换
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # 对细节系数进行阈值处理
    coeffs_thresh = [pywt.threshold(detail, threshold) for detail in coeffs[1:]]
    
    # 重构信号
    data_denoised = pywt.waverec(coeffs_thresh, wavelet)
    
    return data_denoised
