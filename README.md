# quench_detector
Software for processing, characterization, and trigger algorithms for quench detection in training 

`process.sh` takes `.tdms` files with maximum current and ramp argument to produce `.npy` array, each row 'i' such that `data[i]` corresponds to the ordered channel. Current is saved as `a[3]` of acoustic channel. 
