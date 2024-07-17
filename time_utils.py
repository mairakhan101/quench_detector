def calculate_ratios(data, delta_t):
    adjusted_values = [np.roll(data[i+1], -int(delta_t[i])) for i in range(0, len(data[0]), 2)]

    # Calculate the ratios
    ratios = [data[i] / adjusted_values[i//2] for i in range(0, len(data), 2)]

    return ratios

def calculate_rms(ratios):
    return np.sqrt(np.mean(np.square(np.subtract(ratios, 1))))

def find_optimal_delta_t(data, sample_rate=1e6):
    delta_t = [0.0, 0.0]
    
    time_step = 1 / sample_rate
    
    threshold = 0.001
    
    for i in range(2):
        ratios = calculate_ratios(data, delta_t)
        rms = calculate_rms(ratios)
        
        while rms > threshold:
            adjusted_values = [np.roll(data[j+1], -int(delta_t[j])) for j in range(0, len(data), 2)]
            
            current_ratios = [data[j] / adjusted_values[j//2] for j in range(0, len(data), 2)]
            
            rms = calculate_rms(current_ratios)
            
            delta_t[i] += 0.001 * time_step 
    
    return delta_t
