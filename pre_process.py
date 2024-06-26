#Data processing
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile #Process ramping file
import urllib.request
import scipy.io
from scipy.signal import welch
import pywt 
from scipy import signal
import sys
import glob
import nptdms


#PrepProcesss


# Read data from a TDMS file to Pandas DataFrames.
def sma(window, data):
    ave = np.convolve(data, np.ones(window)/window, mode='valid')
    pad_length = len(data) - len(ave)
    padded = np.pad(ave, (0, pad_length), 'constant')
    return padded

def ac(ramp, max_cs):
    file = glob.glob(os.path.join('/home/maira/Magnets/preprocess_data/Acoustics/', '*.tdms'))[0]
    tdms_file = TdmsFile.read(file)
    data_frame = tdms_file.as_dataframe()
    df = data_frame
    current_index = data_frame.keys()
    
    new_columns = ['s0','s1','s2','curr']
    
    df.rename(columns=dict(zip(df.columns, new_columns)), inplace=True)
    data_arr = df.values
    ac_arr = data_arr.T
    x = np.where(ac_arr[3]>=max_cs[ramp-1])
    max_c = x[0][0]
                     
    return data_arr, max_c

def time(data_arr, maxs):
    time_range = np.arange(start = 0, stop = data_arr.shape[0], step = 1, dtype = 'float32')
    time_range -= maxs
    time_range = np.multiply(time_range, 1e-6, out=time_range, casting="unsafe")
    time_range = time_range[0:maxs]
    data_arr = data_arr[:][0:maxs]
    data_arr = data_arr.T
    time = time_range
    return data_arr, time
    

def read_data(path, group_name=None):
    file = TdmsFile(path)
    
    if group_name is None:
        data = {}
        for group in file.groups():
            group_data = {}
            for channel in file[group.name].channels():
                time = channel.time_track()
                channel_data = channel[:]
                group_data[channel.name] = channel_data
            df = pd.DataFrame(group_data, index=time)
            data[group.name] = df
    
    else:
        group_data = {}
        group_name = group_name.replace("'", '')
        for channel in file[group_name].channels():
            time = channel.time_track()
            channel_data = channel[:]
            group_data[channel.name] = channel_data
        data = pd.DataFrame(group_data, index=time)
    
    del file
    return data


# Locate the quench time
def locate_quench(data):
    volt_32 = data['Voltage_32']
    grad = np.gradient(volt_32)
    index = np.argmax(grad)
    time = volt_32.index[index]
    
    del volt_32, grad
    return index, time


# Set the quench time to zero
def reparametrize_time(data, quench_time=None):
    if quench_time is None:
        quench_time = locate_quench(data)[1]
    
    data.index = data.index - quench_time
    
# Read the data of a particular ramp and reparametrize the time
def read_ramp(ramp):

    tdms_file_U = nptdms.TdmsFile.read('/home/maira/Magnets/preprocess_data/QA/QU/AllCards.tdms')
    groups_U = tdms_file_U.groups()
    u_group = groups_U[0].path.split("/")[-1]
    tdms_file_D = nptdms.TdmsFile.read('/home/maira/Magnets/preprocess_data/QA/QD/AllCards.tdms')
    groups_D = tdms_file_D.groups()
    d_group =  groups_D[0].path.split("/")[-1]
    data = {}
    
    data['DAQD'] = read_data('/home/maira/Magnets/preprocess_data/QA/QD/AllCards.tdms', 
                             d_group).astype('float16')
    data['DAQU'] = read_data('/home/maira/Magnets/preprocess_data/QA/QU/AllCards.tdms', 
                             u_group).astype('float16')
    reparametrize_time(data['DAQD'])
    reparametrize_time(data['DAQU'])
    
    return data


def make_qa_array(data, time):
    mini = time[0]
    qd = np.array(data['DAQD'][mini:0.0000])
    qu = np.array(data['DAQU'][mini:0.0000])    
    qd = (qd.T)[0:29]
    qu = np.vstack([(qu.T)[0:24], (qu.T)[25:32]])
    qa = np.vstack([qd, qu])
    return qa 

def make_dict(arr, qa_labels, order):
    '''
    Input is list of channel names, ordered channels and q_array

    '''

    label_to_index = {label: idx for idx, label in enumerate(order)}
    q_dict = {}
    for label in order:
        idx = label_to_index[label]
        indices = [i for i, l in enumerate(qa_labels) if l == label]
        q_dict[label] = arr[indices]

    return q_dict





def main():
    ramp_number = int(sys.argv[1])

    max_curr = [
    12786, 15038, 14820, 11063, 9336, 11329, 11120, 9129, 8967, 9502,
    9416, 9088, 9590, 9659, 9230, 11001, 12254, 12118, 12294, 11042,
    10832, 10220, 10840, 10409, 10871, 10679, 10822, 10363, 10374, 9192,
    9309, 9345, 9809, 10441, 10735, 11066, 10613, 10878, 10861, 8680,
    9410, 13122, 12061, 10724, 10663, 10955, 9617, 9437, 10196, 10700,
    10847, 10842, 10853, 11010, 8926, 9273, 9283]
    
    data_arr, max_c = ac(ramp_number, max_curr)
    ac_arr , t= time(data_arr, int(max_c))

    

    np.save('/home/maira/Magnets/preprocess_data/Acoustics/ac_arr.npy', ac_arr)

    np.save('/home/maira/Magnets/preprocess_data/Acoustics/t_arr.npy', t)

    q_temp = read_ramp(ramp_number)
    q_arr = make_qa_array(q_temp, t)

    qa_labels = [
    'RE_T6', 'RE_T14', 'IN_T4_T17', 'RE_T19', 'IN_T5_T16', 'OUT_T2_T19', 'OUT_T4_T17', 'RE_T12', 'RE_T15',
    'RE_T16', 'OUT_T5_T16', 'OUT_T1_T20', 'OUT_T3_T18', 'IN_T3_T18', 'RE_T18', 'RE_T9', 'IN_T2_T19', 'RE_T11',
    'RE_T13', 'RE_T10', 'RE_T17', 'RE_T20', 'RE_T5', 'RE_T4', 'RE_T1', 'RE_T8', 'RE_T2', 'RE_T3', 'RE_T7', 'LE_T10', 'LE_T12', 'LE_T11',  'LE_T19', 'LE_T2', 'LE_T4', 'LE_T16', 'LE_T13', 'LE_T6', 'LE_T17',
    'LE_T14', 'LE_T9', 'LE_T15', 'LE_T20', 'LE_T3', 'LE_T1', 'LE_T7', 'LE_T18', 'LE_T8', 'LE_T5', 'IN_T9_T12',
    'OUT_T7_T14', 'OUT_T10_T11', 'IN_T10_T11', 'IN_T7_T14', 'OUT_T8_T13', 'IN_T6_T15', 'OUT_T9_T12', 'OUT_T6_T15',
    'IN_T8_T13', 'IN_T1_T20']


order = ['LE_T1', 'RE_T1', 'LE_T2', 'RE_T2', 'LE_T3', 'RE_T3', 'LE_T4', 'RE_T4', 'LE_T5', 'RE_T5',  'LE_T6', 'RE_T6', 'LE_T7', 'RE_T7',  'LE_T8', 'RE_T8',  'LE_T9',  'RE_T9',
         'LE_T10','RE_T10',  'LE_T11', 'RE_T11', 'LE_T12','RE_T12', 'LE_T13','RE_T13','LE_T14','RE_T14', 'LE_T15','RE_T15', 'LE_T16', 'RE_T16', 'LE_T17', 'RE_T17', 'LE_T18',
          'RE_T18',  'LE_T19',  'RE_T19', 'LE_T20', 'RE_T20','IN_T1_T20', 'OUT_T1_T20', 'IN_T2_T19', 'OUT_T2_T19', 'IN_T3_T18', 'OUT_T3_T18',  'IN_T4_T17', 'OUT_T4_T17',
          'IN_T5_T16', 'OUT_T5_T16', 'IN_T6_T15',  'OUT_T6_T15', 'IN_T7_T14',  'OUT_T7_T14', 'IN_T8_T13','OUT_T8_T13', 'IN_T9_T12', 'OUT_T9_T12', 'IN_T10_T11', 'OUT_T10_T11']

    make_dict(qa_arr)

    
    
    np.save('/home/maira/Magnets/preprocess_data/QA/q_arr.npy', q_arr)
    

    
if __name__ == "__main__":
    main()
