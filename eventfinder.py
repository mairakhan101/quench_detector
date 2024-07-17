import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
from tqdm import tqdm
from scipy.fft import fft, ifft
from scipy.stats import describe
from scipy.signal import butter, filtfilt
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt
import mplhep as hep
import mpld3
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

hep.style.use("CMS")

def filter_60Hz_ac(signal, sr):
    signal = signal.astype(np.float64)  # Ensure signal is in float64 for FFT
    signal -= np.mean(signal)
    
    # Compute FFT of the signal
    amp = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/sr)  # Frequency axis with appropriate spacing
    
    # Identify and filter harmonics of 60 Hz (and its multiples)
    harmonics_indices = np.where(np.abs(np.round(freq / 60) * 60 - freq) < 0.2)
    amp[harmonics_indices] = 0
    
    # Reconstruct the filtered signal using inverse FFT
    filtered_signal = np.fft.ifft(amp).real
    
    # Convert back to int16
    filtered_signal = filtered_signal[:len(signal)].astype(np.int16)  # Truncate to original length and convert to int16
    
    return filtered_signal

def filter_60Hz_qa(signal, sr):
    # Remove DC offset by subtracting the mean
    signal -= np.mean(signal)
    
    # Compute FFT of the signal
    amp = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/sr)  # Frequency axis with appropriate spacing
    
    # Identify and filter harmonics of 60 Hz (and its multiples)
    harmonics_indices = np.where(np.abs(np.round(freq / 60) * 60 - freq) < 0.5)
    amp[harmonics_indices] = 0
    
    # Reconstruct the filtered signal using inverse FFT
    filtered_signal = np.fft.ifft(amp).real
    
    # Convert back to int16
    filtered_signal = filtered_signal[:len(signal)]  
    
    return filtered_signal

def filter_freq_ac(signal, frequencies_to_filter, sr):
    signal = signal.astype(np.float64)
    signal -= np.mean(signal)
    
    amp = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/sr)  
    
    for freq_to_filter in frequencies_to_filter:
        indices = np.where(np.abs(freq - freq_to_filter) < 0.3)
        amp[indices] = 0
    
    filtered_signal = np.fft.ifft(amp).real
    
    filtered_signal = filtered_signal[:len(signal)].astype(np.int16)  
    
    return filtered_signal

def filter_freq_qa(signal, frequencies_to_filter, sr):
    # Remove DC offset by subtracting the mean
    signal -= np.mean(signal)
    
    # Compute FFT of the signal
    amp = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/sr)  # Frequency axis with appropriate spacing
    
    # Identify and filter specified frequencies
    for freq_to_filter in frequencies_to_filter:
        indices = np.where(np.abs(np.round(freq /freq_to_filter) - freq) < 1)
        amp[indices] = 0
    
    # Reconstruct the filtered signal using inverse FFT
    filtered_signal = np.fft.ifft(amp).real
    filtered_signal = filtered_signal[:len(signal)]  
    
    return filtered_signal

def plot_fft(signal, sr, max_freq):
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/sr)  
    mask = (freq >= 0) & (freq <= max_freq)
    freq = freq[mask]
    fft_result = fft_result[mask]
    magnitude = np.abs(fft_result)
    normalized_magnitude = magnitude / np.sqrt(len(signal))  

    threshold = 0.02

    for i in range(len(freq)):
        if normalized_magnitude[i] > threshold:
            plt.text(freq[i], normalized_magnitude[i], f'{freq[i]:.1f} Hz', fontsize=10, ha='center', va='bottom')

    plt.plot(freq, normalized_magnitude)
    plt.title(f'FFT Magnitude Normalized (0 to {max_freq} Hz)', fontsize = 20)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.show()

def get_peaks(filt_signal, window, threshold):
    smoothed_signal = np.convolve(signal, np.ones(window)/window, mode='same')
    peaks, _ = find_peaks(np.abs(smoothed_signal), height =threshold)
    print(f'Num peaks {len(peaks)}')
    return smoothed_signal, peaks 


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
        end_index = min(peaks[j-1] + end // 2, len(t_q))
    
        # Store the start and end indices
        start_indices.append(start_index)
        end_indices.append(end_index)
    
        # Move to the next group of peaks
        i = j
    return start_indices, end_indices


def main():
    ramp = int(sys.argv[1])
    a = np.load(f'/home/maira/Magnets/preprocess_data/ac_arr_r{ramp}.npy')
    q = np.load(f'/home/maira/Magnets/preprocess_data/q_arr_r{ramp}.npy')
    t = np.load(f'/home/maira/Magnets/preprocess_data/t_arr_r{ramp}.npy')
    t_q = t[::10]

    qa_labels = [
    'RE_T6', 'RE_T14', 'IN_T4_T17', 'RE_T19', 'IN_T5_T16', 'OUT_T2_T19', 'OUT_T4_T17', 'RE_T12', 'RE_T15',
    'RE_T16', 'OUT_T5_T16', 'OUT_T1_T20', 'OUT_T3_T18', 'IN_T3_T18', 'RE_T18', 'RE_T9', 'IN_T2_T19', 'RE_T11',
    'RE_T13', 'RE_T10', 'RE_T17', 'RE_T20', 'RE_T5', 'RE_T4', 'RE_T1', 'RE_T8', 'RE_T2', 'RE_T3', 'RE_T7', 'LE_T10', 'LE_T12', 'LE_T11',   'LE_T19', 'LE_T2',  'LE_T4', 'LE_T16', 'LE_T13', 'LE_T6', 'LE_T17',
    'LE_T14', 'LE_T9', 'LE_T15', 'LE_T20', 'LE_T3', 'LE_T1', 'LE_T7', 'LE_T18', 'LE_T8', 'LE_T5', 'IN_T9_T12',
    'OUT_T7_T14', 'OUT_T10_T11', 'IN_T10_T11', 'IN_T7_T14', 'OUT_T8_T13', 'IN_T6_T15', 'OUT_T9_T12', 'OUT_T6_T15',
    'IN_T8_T13', 'IN_T1_T20']

    order = ['LE_T1', 'RE_T1', 'LE_T2', 'RE_T2', 'LE_T3', 'RE_T3', 'LE_T4', 'RE_T4', 'LE_T5', 'RE_T5',  'LE_T6', 'RE_T6', 'LE_T7', 'RE_T7',  'LE_T8', 'RE_T8',  'LE_T9',  'RE_T9',
         'LE_T10','RE_T10',  'LE_T11', 'RE_T11', 'LE_T12','RE_T12', 'LE_T13','RE_T13','LE_T14','RE_T14', 'LE_T15','RE_T15', 'LE_T16', 'RE_T16', 'LE_T17', 'RE_T17', 'LE_T18',
          'RE_T18',  'LE_T19',  'RE_T19', 'LE_T20', 'RE_T20','IN_T1_T20', 'OUT_T1_T20', 'IN_T2_T19', 'OUT_T2_T19', 'IN_T3_T18', 'OUT_T3_T18',  'IN_T4_T17', 'OUT_T4_T17',
          'IN_T5_T16', 'OUT_T5_T16', 'IN_T6_T15',  'OUT_T6_T15', 'IN_T7_T14',  'OUT_T7_T14', 'IN_T8_T13','OUT_T8_T13', 'IN_T9_T12', 'OUT_T9_T12', 'IN_T10_T11', 'OUT_T10_T11']
    curr = ['s0', 's1', 's2', 'curr']

    import pickle
    for i in range(16, 60):
        print(i)
        signal = q[i]
        name = qa_labels[i]
        q_filt = filter_60Hz_qa(q[i], 1e5) 
        q_filt = filter_freq_qa(q_filt, [43.4, 108.8], 1e5)
        #plot_fft(q_filt, 1e5, 1000)
        #plot_fft(q[i], 1e5, 1000)
        #plt.plot(q[i][0:10000])
        smoothed_signal, peaks = get_peaks(q_filt, 100, 0.0008)
        start_indices, end_indices = get_indices(smoothed_signal, peaks, 2000, 1000, 4000)
    
        with PdfPages(f'qa_events{name}_{ramp}.pdf') as pdf:
            t_qs = []
            smooths = []
            raws = []
        
            for j in range(0, len(start_indices)):
                start = start_indices[j]
                end = end_indices[j]
                t_qs.append(t_q[start:end])
                smooths.append(smoothed_signal[start:end])
                raws.append(signal[start:end])

                plt.figure(figsize=(20, 6))  # Adjust figure size as needed
                # Plot smoothed signal
                plt.subplot(1, 3, 1)  # Subplot 1: smoothed signal
                plt.plot(t_q[start:end], smoothed_signal[start:end], label='Smoothed Signal')
                plt.xlabel('Time Before Quench (s)', fontsize=14)
                plt.ylabel('Voltage Rise (V)', fontsize=14)
                plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Set major ticks every 0.1 units
                plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f')) 
                plt.tick_params(axis='x', labelsize=12)
                plt.tick_params(axis='y', labelsize=12)
                plt.title(f'Event {j} (Smoothed)', fontsize=20)
                plt.grid(True)
    
                # Plot normal signal
                plt.subplot(1, 3, 2)  # Subplot 2: normal signal
                plt.plot(t_q[start:end], signal[start:end], label='Normal Signal')
                plt.xlabel('Time Before Quench (s)', fontsize=14)
                plt.ylabel('Voltage Rise (V)', fontsize=14)
                plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Set major ticks every 0.1 units
                plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                plt.tick_params(axis='x', labelsize=12)
                plt.tick_params(axis='y', labelsize=12)
                plt.title(f'Event {j} (Raw Filtered)', fontsize=20)
                plt.grid(True)

                plt.tight_layout()

                pdf.savefig()
                plt.close()  # Close the current figure to free up memory

        # Save processed data as pickle files
        with open(f'/home/maira/Magnets/preprocess_data/events/qa_{ramp}_{name}_t.pkl', 'wb') as f:
            pickle.dump(t_qs, f)
        with open(f'/home/maira/Magnets/preprocess_data/events/qa_{ramp}_{name}_smooth.pkl', 'wb') as f:
            pickle.dump(smooths, f)
        with open(f'/home/maira/Magnets/preprocess_data/events/qa_{ramp}_{name}_raw.pkl', 'wb') as f:
            pickle.dump(raws, f)

        plt.close('all')  # Close all figures at the end of each iteration
    

    
if __name__ == "__main__":
    main()