from EMG_burst import EMG_to_binary_signal
import numpy as np
import scipy as sp
import scipy.signal # Some version needs it
import scipy.stats  # Some version needs it
import pandas as pd
import matplotlib.pyplot as plt

tremogram = pd.read_csv('./test.csv')
signal = tremogram[tremogram.columns[1]].to_numpy()
time = tremogram['Time'].to_numpy()

output = EMG_to_binary_signal(signal, time, [2.94,3.01], Fs=5000, ma_dur=40E-3, thr_sig=15, bp_filt=True, bp_filt_low=10, bp_filt_high=300)

plt.plot(time, output['processed_signal'], time, output['binary_signal'])
plt.show()

plt.plot(time, output['processed_signal'], time, output['binary_signal'])
plt.xlim([30,31])
plt.show()
