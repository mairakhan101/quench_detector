#Data processing
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import scipy.io
import numpy as np
from scipy import signal
import wandb
import random

ac0 = np.load('/home/maira/Magnets/code_revamp/tmp/ai0.npy')
ac1 =  np.load('/home/maira/Magnets/code_revamp/tmp/ai1.npy')
ac2 = np.load('/home/maira/Magnets/code_revamp/tmp/ai2.npy')
ac3 = np.load('/home/maira/Magnets/code_revamp/tmp/ai3.npy')
ac4 = np.load('/home/maira/Magnets/code_revamp/tmp/ai4.npy')
curr =  np.load('/home/maira/Magnets/code_revamp/tmp/curr.npy')
t = np.load('/home/maira/Magnets/code_revamp/tmp/time.npy')

def normalized(signal):
    mean_val = np.mean(signal)
    normalized_signal = signal - mean_val 
    return normalized_signal

def get_peaks(filt_signal, window, threshold):
    peaks, _ = find_peaks(np.abs(filt_signal), height =threshold)
    print(f'Num peaks {len(peaks)}')
    return peaks 


def get_indices(smoothed_signal, peaks, window_size, peak_dist, end):
    start_indices = []
    end_indices = []

    i = 0
    while i < len(peaks):
        peak_index = i
        start_index = max(0, peaks[peak_index] - window_size // 2)
    
        # Find the end of the window where consecutive peaks are within 10 units
        j = i + 1
        while j < len(peaks) and peaks[j] - peaks[j-1] <= peak_dist:
            j += 1
    
        # Set the end index of the window
        end_index = min(peaks[j-1] + end // 2, len(smoothed_signal))
    
        # Store the start and end indices
        start_indices.append(start_index)
        end_indices.append(end_index)
    
        # Move to t3e next group of peaks
        i = j
    return start_indices, end_indices

import pickle
from scipy.signal import find_peaks

for i in range(0,5):
    time = t
    current = curr
    sig_name = f'ac{i}'
    signal = globals()[sig_name]
    fs = 100000
    f0 = fs / 1667
    Q = 30
    [b,a] = scipy.signal.iircomb(f0, Q, ftype='notch', fs=fs)
    signal -= np.mean(signal)
    signal_filt = scipy.signal.lfilter(b,a,signal)
    ac_filt = normalized(signal_filt)
    peaks = get_peaks(ac_filt, 100, 0.1)
    start_indices, end_indices = get_indices(ac_filt, peaks, 2000, 1000, 4000)
    #Save local array as time, current singal 
        
    for j, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
        event_signal = ac_filt[start_idx:end_idx]
        t_signal = time[start_idx:end_idx]
        curr_signal = current[start_idx:end_idx]
        # Stack time, current, event 
        event_array = np.vstack((t_signal, curr_signal, event_signal))  # Here current == signal, modify if needed
        filename = f"r_event_{j}_ac_{i}_r1.npy"
        if time[start_idx] > -0.2:
            file_path = os.path.join('/home/maira/Magnets/code_revamp/precursors/', filename)
            np.save(file_path, event_array)
        else: 
            file_path = os.path.join('/home/maira/Magnets/code_revamp/early_events/', filename)
            np.save(file_path, event_array)
        if j<int(len(start_indices)-1):
            noise = ac_filt[end_indices[j]:start_indices[j+1]]
            noise_curr = current[end_indices[j]:start_indices[j+1]]
            noise_time = time[end_indices[j]:start_indices[j+1]]
            noise_array = np.vstack((noise_time, noise_curr, noise))
            filename = f"r_noise_{j}_ac_{i}_r1.npy"
            file_path = os.path.join('/home/maira/Magnets/code_revamp/noise/', filename)
            np.save(file_path, noise_array)
        
    
    
