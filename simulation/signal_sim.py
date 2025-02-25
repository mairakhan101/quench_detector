import numpy as np
from scipy.fft import fft, ifft
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import glob
import os

def exp_decay(t, A0, alpha):
    """Exponential decay function for fitting."""
    return A0 * np.exp(-alpha * t)

def fit_exponential_envelope(signal, sample_rate):
    """Fits an exponential decay to the envelope of a time-domain signal."""
    time = np.arange(len(signal)) / sample_rate
    peak_idx = np.argmax(np.abs(signal))  # Find peak location
    peak_value = np.abs(signal[peak_idx])  # Get max amplitude
    diffs = np.diff(signal)
    sorted_indices = np.argsort(np.abs(diffs))[::-1]
    sorted_diff_indices = sorted_indices[:len(diffs)]  

    peaks = []
    for i in range(0, len(sorted_diff_indices)): 
        if np.abs(sorted_diff_indices[i] - peak_idx) > 200:
            if sorted_diff_indices[i] > peak_idx:
                peaks.append(sorted_diff_indices[i])
    
    A0 = peak_value
    if len(peaks)>0:
        end_indx = peaks[0]
    else: 
        end_indx = len(signal)
        # fix this condtion 
        
    # Fit the decay using data after the peak
    t_decay = time[peak_idx:end_indx]
    envelope = np.abs(signal[peak_idx:end_indx])

    # Check if the decay function starts from the peak value
    if np.isclose(envelope[0], A0, atol=1e-5):  # Condition to ensure the max is included
        try:
            popt, _ = curve_fit(exp_decay, t_decay, envelope, p0=(A0, 0.001))
        except RuntimeError:
            popt = [A0, 0.001]  # Fallback parameters if fit fails
    else:
        popt = [A0, 0.001]  # If no fit possible, use the peak value as A0 and alpha as a default value
    
    return popt, peak_idx, end_indx 

def generate_synthetic_acoustic(signal, sample_rate):
    """
    Generate synthetic acoustic signals using FFT and an exponential decay model.
    
    Also fits an exponential decay to the time-domain envelope and reintroduces
    residual noise at a random point.
    """
    # FFT of the input signal
    freq_domain = fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)

    # Consider only positive frequencies (excluding f=0)
    pos_mask = (freqs > 0)
    pos_freqs = freqs[pos_mask]
    magnitude = np.abs(freq_domain[pos_mask])

    # Fit an exponential decay to the magnitude spectrum
    try:
        popt_freq, _ = curve_fit(exp_decay, pos_freqs[1:], magnitude[1:], p0=(magnitude[0], 0.001))
    except RuntimeError:
        popt_freq = [magnitude[0], 0.001]

    # Generate new magnitudes using the fitted model
    new_magnitude = exp_decay(pos_freqs, *popt_freq)

    # Assign random phases
    random_phase = np.exp(1j * np.random.uniform(-np.pi, np.pi, len(new_magnitude)))

    # Construct new frequency-domain representation
    new_freq_domain = np.zeros_like(freq_domain, dtype=complex)
    new_freq_domain[pos_mask] = new_magnitude * random_phase
    new_freq_domain[-len(new_magnitude):] = np.conj(new_freq_domain[1:len(new_magnitude) + 1][::-1])

    # IFFT to get the new time-domain signal
    new_signal = np.real(ifft(new_freq_domain))
    new_signal_centered = new_signal - np.mean(new_signal) 
    max_val = np.max(np.abs(new_signal))  # Find the maximum absolute value
    ns = new_signal_centered / max_val  # Scale to [-1, 1]
    # Fit an exponential decay to the signal envelope
    popt_env, peak_idx, end_indx = fit_exponential_envelope(signal, sample_rate)
    A0 = np.max(signal)  
    popt_env = [A0, popt_env[1]]
    # Generate the modeled decay envelope
    time = np.arange(len(signal[np.argmax(signal):end_indx])) / sample_rate
    decay_model = exp_decay(time, *popt_env)
    pre_max = ns[0:np.argmax(signal)]
    pre_max = pre_max*2*np.sqrt(np.mean(np.square(signal[0:np.argmax(signal)])))
    sim = decay_model*2*ns[np.argmax(signal):end_indx] + ns[np.argmax(signal):end_indx]*np.sqrt(np.mean(np.square(signal[np.argmax(signal):end_indx])))
    post_sig = ns[end_indx:]*2*np.sqrt(np.mean(np.square(signal[end_indx:])))
    sim_sig = np.concatenate((pre_max, sim, post_sig))

    
    plt.plot(signal)
    plt.plot(sim_sig)
    
    actual_integral = np.sum(np.abs(signal))
    simulated_integral = np.sum(np.abs(sim_sig))
    residual_energy = actual_integral - simulated_integral
    print(actual_integral)
    #
    if simulated_integral > actual_integral:
        scale_factor = 0.99  # Start with 99% and decrease gradually

        while simulated_integral > actual_integral and residual_energy < -1:
            sim_sig[0:end_indx] *= scale_factor  # Scale down
            simulated_integral = np.sum(np.abs(sim_sig))  # Recalculate
            residual_energy = actual_integral - simulated_integral
            print(f"Scaling down: New Integral = {simulated_integral}, Residual = {residual_energy}")

        if residual_energy >= -2:  # Stop once within acceptable range
            print("Simulated signal scaled down successfully.")
            return sim_sig 
    
    # Choose ratio of Log-Normal vs White Noise
    log_normal_ratio = np.random.uniform(0.3, 0.7)  # 30-70% log-normal noise
    white_noise_ratio = 1 - log_normal_ratio

    # Generate Log-Normal Noise (Spiky)
    num_samples = len(sim_sig)
    mean_amplitude = np.mean(np.abs(sim_sig)) + 1e-6  # Avoid log(0)
    
    sigma = 1.2  # Higher sigma = more spiky log-normal noise
    lognormal_noise = np.random.lognormal(mean=np.log(mean_amplitude), sigma=sigma, size=num_samples)
    
    # Make it symmetric
    lognormal_noise *= np.random.choice([-1, 1], size=num_samples)

    # Generate White Noise Using RMS of a Specific Section
    rms_window = signal[-int(0.1 * len(signal)):]  # Last 10% of the signal
    rms_value = np.sqrt(np.mean(np.square(rms_window)))  # Compute RMS

    white_noise = np.random.normal(loc=0, scale=rms_value, size=num_samples)

    # Normalize noise to match residual energy
    lognormal_noise *= log_normal_ratio
    white_noise *= white_noise_ratio

    # Compute total absolute energy and rescale
    total_noise = lognormal_noise + white_noise
    total_noise_energy = np.sum(np.abs(total_noise))

    if total_noise_energy == 0:
        print("Generated noise sum is zero, check parameters.")
        return sim_sig

    # Initial Scaling Correction
    total_noise *= residual_energy / total_noise_energy  # Ensures sum of |noise| = residual

    # Add to the simulated signal
    adjusted_signal = sim_sig + total_noise

    # **Final Residual Correction Step**
    final_residual = np.sum(np.abs(signal)) - np.sum(np.abs(adjusted_signal))
    print(f"Pre-Final Correction Residual: {final_residual}")  

    # Only scale if residual is still outside acceptable range
    epsilon = 0.1
    if abs(final_residual) > epsilon:
        correction_factor = actual_integral / np.sum(np.abs(adjusted_signal))
        adjusted_signal *= correction_factor  # Scale to match within epsilon

    # Final Check: Residual Should Be **Within ε**
    final_residual = np.sum(np.abs(signal)) - np.sum(np.abs(adjusted_signal))
    print(f"Final Residual: {final_residual} (Target: within ±{epsilon})")  

    return adjusted_signal
    
    
# Example Usage
if __name__ == "__main__":
    sample_rate = 1e5  # Hz
    num_samples = 200
    e_dir = '/home/maira/Magnets/code_revamp/early_events/'
    p_dir = '/home/maira/Magnets/code_revamp/precursors/'
    
    
    
    npy_files = glob.glob(os.path.join(p_dir, '*.npy'))
    for file_path in npy_files:
        sig = np.load(file_path)
        time = sig[0]
        curr = sig[1]
        original_signal = sig[2]
        synthetic_signal = generate_synthetic_acoustic(original_signal, sample_rate)

        # Plot for comparison
        plt.figure(figsize=(10, 5))
        plt.plot(time, original_signal, label="Original Signal", alpha=0.7)
        plt.plot(time, synthetic_signal, label="Synthetic Signal", linestyle="dashed")
        plt.legend()
        plt.show()
