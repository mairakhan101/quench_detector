from scipy.signal import welch
import numpy as np
import sys
from tqdm import tqdm
from scipy.fft import fft, ifft
from scipy.stats import describe
from scipy.signal import butter, filtfilt
from scipy.signal import firwin, lfilter

def filter_60Hz(signal):
    amp = np.fft.fft(signal)
    freq = np.round(np.fft.fftfreq(len(signal), d=1e-5) / 10) * 10
    
    indices = np.where((freq % 60 == 0) & (freq != 0))
    amp[indices] = 0
    filtered_signal = np.fft.ifft(amp).real
    #indices = np.where(freq >= 0)
    #filtered_fft = pd.Series(np.abs(amp[indices]), index=freq[indices])
    return filtered_signal



def spectral_entropy(signal):
    fft_result = fft(signal)
    psd = np.abs(fft_result) ** 2
    psd /= psd.sum() 
    psd = psd + 1e-6
    entropy = -np.sum(psd * np.log2(psd))
    return entropy



def windowed_se(signal, time, curr, window_size, step):
    
    num_channels, num_samples = signal.shape
    num_filt_samples = (num_samples - window_size) // step + 1
    
    if curr[0] == 0:
        curr_timestamps = 0
        
    else: 
        curr_timestamps = np.zeros(num_filt_samples)
        for i in range(0, num_samples - window_size + 1, step):
            curr_timestamps[i//step] = curr[i + window_size // 2]
        
        
    
    
    window_timestamps = np.zeros(num_filt_samples)
    spectral_entropies = np.zeros((num_channels, num_filt_samples))

    for i in range(0, num_samples - window_size + 1, step):
        window_timestamps[i//step] = time[i + window_size // 2] 
    
    for i in tqdm(range(num_channels), desc="Processing Channels"):
        for j in range(0, num_samples - window_size + 1, step):
            window_signal = signal[i, j:j+window_size]
            spectral_entropies[i, j//step] = spectral_entropy(window_signal)
    
    return spectral_entropies, window_timestamps, curr_timestamps
    
def windowed_stats(signal, time, window_size=200, step=10):
    num_channels, num_samples = signal.shape
    num_filt_samples = (num_samples - window_size) // step + 1
    window_timestamps = np.zeros(num_filt_samples)
    feature_array = np.zeros((num_channels, 3, num_filt_samples)) 
    for i in range(0, num_samples - window_size + 1, step):
        window_timestamps[i//step] = time[i + window_size // 2] 
    
    for i in tqdm(range(num_channels), desc="Processing Channels"):
        for j in range(0, num_samples - window_size + 1, step):
            window_signal = signal[i, j:j+window_size]
            feature_array[i, :, j//step] = [desc.mean, desc.variance, desc.maxmax]
    
    return feature_array, window_timestamps

def main():
    ramp = int(sys.argv[1])

    a = np.load(f'/home/maira/Magnets/preprocess_data/ac_arr_r{ramp}.npy')
    q = np.load(f'/home/maira/Magnets/preprocess_data/q_arr_r{ramp}.npy')
    t = np.load(f'/home/maira/Magnets/preprocess_data/t_arr_r{ramp}.npy')

    curr = a[3]

    q_t = t[::10]
    q_t = q_t[0:q[0].shape[0]]
    
    
    

    #qa = filter_60Hz(q, 1e5)
    #ac = filter_60Hz(a[0:2], 1e6)

    q_se, tq, c_q = windowed_se(q, q_t, [0], 200, 50)
    a_se, ta, c_t = windowed_se(a[0:3],t, curr, 2000, 500)


    #q_stats, t_q = windowed_stats(qa, q_t, 200, 10)
    #a_stats, t_a = windowed_stats(ac, t, 2000, 100)

    if (ramp == 1):
        print(a_se.shape)
        print(q_se.shape)
        print(c_t.shape)
        print(curr.shape)
        print(ta.shape)
        print(tq.shape)
    
    np.save(f'/home/maira/Magnets/preprocess_data/wc_{ramp}.npy', c_t)
    np.save(f'/home/maira/Magnets/preprocess_data/curr_{ramp}.npy', curr)

    np.save(f'/home/maira/Magnets/preprocess_data/filt_q_se_{ramp}.npy', q_se)
    np.save(f'/home/maira/Magnets/preprocess_data/filt_a_se_{ramp}.npy', a_se)

    #np.save(f'/home/maira/Magnets/preprocess_data/filt_q_stats_{ramp}.npy', q_stats)
    #np.save(f'/home/maira/Magnets/preprocess_data/filt_a_stats_{ramp}.npy', a_stats)

    np.save(f'/home/maira/Magnets/preprocess_data/a_t_{ramp}.npy', ta)
    np.save(f'/home/maira/Magnets/preprocess_data/q_t_{ramp}.npy', tq)
    
    
if __name__ == "__main__":
    main()
