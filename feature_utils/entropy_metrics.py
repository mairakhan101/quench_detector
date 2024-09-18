def get_trans_en(ts1, ts2, window, step, time, k):

    num_win = (ts1.shape[0] - window) // step + 1

    te_values = np.empty(num_win)
    window_centers = np.empty(num_win, dtype=int)

    for i in range(num_win):
        window_start = i * step
        window_end = window_start + window

        # Extract the current window
        window_A = np.abs(ts1[window_start:window_end])
        window_B = np.abs(ts2[window_start:window_end])

        # Calculate transfer entropy for the current window
        te_value = transferentropy.transfer_entropy(window_A, window_B, k)

        # Store the result and the center of the window
        te_values[i] = te_value
        window_centers[i] = (window_start + window_end - 1) // 2

        win = window_centers 
        
        for i in range(win.shape[0]): 
            times[i] = time[win[i]]

    return te_values, window_centers
