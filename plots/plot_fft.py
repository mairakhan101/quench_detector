def plot_filt_fft(signal, sample_rate, mag_threshold, max_freq=None):
    signal -= np.mean(signal)  # Remove DC component
    fft_result = np.fft.fft(signal)
    N = len(signal)
    freq = np.fft.fftfreq(N, d=1/sample_rate)
    magnitude_spectrum = np.abs(fft_result)
    
    # Normalize to max magnitude
    max_magnitude = np.max(magnitude_spectrum)
    norm_fft = magnitude_spectrum / max_magnitude

    # Apply max_freq if specified
    if max_freq is not None:
        max_freq_index = np.argmax(freq > max_freq)
        freq = freq[:max_freq_index]
        norm_fft = norm_fft[:max_freq_index]

    # Filter frequencies to include only positive values and those above mag_threshold
    positive_freq_mask = (freq > 0) & (freq < 1000)  # Adjust 1000 to your desired minimum frequency
    freq = freq[positive_freq_mask]
    norm_fft = norm_fft[positive_freq_mask]

    # Initialize arrays to store filtered frequencies and magnitudes
    filtered_freq = []
    filtered_norm_fft = []

    # Iterate over frequencies to handle peaks within correction error range
    i = 0
    while i < len(freq):
        if norm_fft[i] > mag_threshold:
            # Check if the peak is within correction error range of previous peak
            found_peak = False
            for j in range(len(filtered_freq)):
                if np.abs(freq[i] - filtered_freq[j]) <= 0.01:
                    # Compare magnitudes instead of frequencies
                    if norm_fft[i] > filtered_norm_fft[j]:
                        filtered_freq[j] = freq[i]
                        filtered_norm_fft[j] = norm_fft[i]
                    found_peak = True
                    break
            if not found_peak:
                filtered_freq.append(freq[i])
                filtered_norm_fft.append(norm_fft[i])
        i += 1

    # Plot FFT spectrum
    plt.figure(figsize=(6, 6))
    plt.plot(freq, norm_fft, label='FFT Spectrum')

    # Label filtered peaks above threshold and plot red dots
    for i in range(len(filtered_freq)):
        plt.text(filtered_freq[i], filtered_norm_fft[i], f'{filtered_freq[i]:.3f} Hz', fontsize=9, ha='center', va='bottom')
        plt.scatter(filtered_freq[i], filtered_norm_fft[i], color='red', s=10, zorder=5)

    # Customize plot
    plt.title(f'FFT Spectrum of {channel_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.tight_layout()
    plt.show()


def plot_frequency_component(channel_name, q_dict, sample_rate, target_frequency, frequency_tolerance=0.5, time=None):
    # Extract signal from q_dict based on channel_name
    signal = q_dict[channel_name][0]
    signal -= np.mean(signal)  # Remove DC component

    #plt.plot(signal[::100000], time[::100000])

    # Compute FFT
    fft_result = np.fft.fft(signal)
    N = len(signal)
    freq = np.fft.fftfreq(N, d=1/sample_rate)

    # Find indices within ±frequency_tolerance Hz range around target_frequency
    index_min = np.argmax(freq >= (target_frequency - frequency_tolerance))
    index_max = np.argmax(freq >= (target_frequency + frequency_tolerance))

    # Filter frequencies within ±frequency_tolerance Hz range around target_frequency
    fft_result_filtered = np.zeros_like(fft_result)
    fft_result_filtered[index_min:index_max] = fft_result[index_min:index_max]

    # Apply IFFT to get time domain signal
    reconstructed_signal = np.fft.ifft(fft_result_filtered)

    # Calculate number of samples for two cycles of target_frequency
    period_samples = int(sample_rate / target_frequency)
    total_samples = 10 * period_samples

    # Extract two cycles of the reconstructed signal
    reconstructed_signal = reconstructed_signal[len(reconstructed_signal)-total_samples: len(reconstructed_signal)]

    # Plot time domain signal
    plt.figure(figsize=(10, 4))
    if time is not None:
        plt.plot(time[len(reconstructed_signal)-total_samples:len(reconstructed_signal)], reconstructed_signal.real, label=f'Reconstructed {target_frequency} Hz Signal ±{frequency_tolerance} Hz')
    else:
        time = np.arange(len(reconstructed_signal)) / sample_rate
        plt.plot(time, reconstructed_signal.real, label=f'Reconstructed {target_frequency} Hz Signal ±{frequency_tolerance} Hz')
    plt.title(f'Time Domain Signal of {channel_name} - {target_frequency} Hz Component at Quench')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



