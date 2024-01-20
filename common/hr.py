import numpy as np
from scipy import signal
import neurokit2 as nk
from enum import Enum

class CalType(Enum):
    FFT = 1
    PEAK = 2
    HRV = 3
class HearRate:
    def __nearest_power_of_2(self,x):
        # Calculate the nearest power of 2.
        return 1 if x == 0 else 2 ** (x - 1).bit_length()
    """
        Calculate heart rate based on PPG using Fast Fourier transform (FFT).
    """
    def calculate_hr(self,ppg_signal, fs=35., low_pass=0.75, high_pass=2.5,cal_type:CalType=CalType.FFT):
        if cal_type == CalType.FFT:
            ppg_signal = np.expand_dims(ppg_signal, 0)
            N = self.__nearest_power_of_2(ppg_signal.shape[1])
            f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
            fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
            mask_ppg = np.take(f_ppg, fmask_ppg)
            mask_pxx = np.take(pxx_ppg, fmask_ppg)
            hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
        if cal_type == CalType.PEAK:
            ppg_peaks, _ = signal.find_peaks(ppg_signal)
            hr = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
        else:
            hrv = self.get_hrv(ppg_signal, fs=fs)
            hr = np.mean(hrv, dtype=np.float32)
        return hr
    def get_hrv(self, ppg_signal, fs=30.):
        ppg_peaks = nk.ppg_findpeaks(ppg_signal, sampling_rate=fs)['PPG_Peaks']
        hrv = nk.signal_rate(ppg_peaks, sampling_rate=fs, desired_length=len(ppg_signal))
        return hrv