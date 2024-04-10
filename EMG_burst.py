## Import everything useful
import numpy as np
import scipy as sp
import scipy.signal # Some version needs it
import scipy.stats  # Some version needs it

import pandas as pd

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def complete_abs_ma(x,w):
    tmp = moving_average(np.absolute(x), w)
    tmp = np.insert(tmp,0,np.zeros(int(np.floor(w/2))-1))
    tmp = np.append(tmp,np.zeros(int(np.floor(w/2))))
    return tmp

def EMG_to_binary_signal(signal, time, ref, Fs, ma_dur=50E-3, thr_sig=1, bp_filt=True, bp_filt_low=10, bp_filt_high=300):
    
    # Optional pre-filtering
    if bp_filt == True:
        SEMG_filt_b, SEMG_filt_a = sp.signal.butter(3,[int(bp_filt_low),int(bp_filt_high)],'bandpass', fs=Fs)
        signal = signal - np.average(signal)
        signal = sp.signal.filtfilt(SEMG_filt_b, SEMG_filt_a, signal)
    
    # Calculate TKEO and rectified_smoothed TKEO
    tkeo = np.power(signal[1:-1],2) - (signal[2:] * signal[0:-2])
    tkeo = np.insert(tkeo, 0, signal[0])
    tkeo = np.append(tkeo, signal[-1])
    rect_smo_tkeo = complete_abs_ma(tkeo, int(Fs*ma_dur))
    
    # Thresholding
    threshold = ((np.std(rect_smo_tkeo[(time > ref[0]) & (time < ref[1])])) * thr_sig) + np.average(rect_smo_tkeo[((time > ref[0]) & (time < ref[1]))])
    
    # Binary output signal
    binary_signal = []
    for i in range(0, len(rect_smo_tkeo)):
        if rect_smo_tkeo[i] >= threshold:
            binary_signal.append(1)
        else:
            binary_signal.append(0)
    return {'binary_signal': binary_signal, 'processed_signal': signal, 'tkeo': tkeo, 'rect_smo_tkeo': rect_smo_tkeo}