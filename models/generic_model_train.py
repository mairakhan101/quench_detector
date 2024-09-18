import numpy as np
import os
import pywt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
import torch.optim as optim
import math
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Transformer
from scipy.signal import cwt
from scipy.signal import cwt, ricker
import nolds
import torch.nn.init as init

# Sample unsupervised ML model 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(60, 128),
            nn.ReLU(),  # You can use any activation function here if desired
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 60),
            nn.ReLU()
        )

        # Initialize weights using Glorot uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#ramps = []
for k in []:
    a = np.load(f'/home/maira/Magnets/preprocess_data/filt_q_se_{k}.npy')
    t = np.load(f'/home/maira/Magnets/preprocess_data/q_t_{k}.npy')
    c = np.load(f'/home/maira/Magnets/preprocess_data/wc_{k}.npy')

    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return self.data.shape[1]

        def __getitem__(self, idx):
            sample = self.data[:, idx]
            return torch.tensor(sample, dtype=torch.float)



    # Create dataset and dataloader, inccluding raw_data
    dataset = CustomDataset(a)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    
    # Initialize model
    model = Autoencoder()

    # Define loss function (reconstruction loss)
    criterion = nn.L1Loss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Gradient clipping
    max_grad_norm = 1.0  # Maximum gradient norm
    clip_grad = torch.nn.utils.clip_grad_norm_

    # Lists to store loss values and latent space representations for plotting
    losses = []
    la = []
    print(f'Training ramp {k}')
    
    for i, batch in enumerate(dataloader):
        input_data = batch
    
    
        input_data = input_data.to(dtype=torch.float32)

        output = model(input_data)
        la.append(output)

        loss = criterion(output, input_data)
    
        # Log the loss
        losses.append(loss.item())
    
    

    
        if (i + 1) % 80000 == 0:
            optimizer.zero_grad()
            loss.backward()
            clip_grad(model.parameters(), max_grad_norm)  # Gradient clipping
            optimizer.step()

            
    print("Done training")
            
    torch.cuda.empty_cache()
    num_sections = 100 #choose sections
    l = np.array(losses)
    section_size = len(l) // num_sections

    for j in range(num_sections):
        section_start = j * section_size
        section_end = section_start + section_size
        l_section = l[section_start:section_end]
        t_section = t[section_start:section_end]
        c_section = c[section_start:section_end]

        rms_section = np.sqrt(np.mean(l_section ** 2))
        std_dev_section = np.std(l_section)

        if j<=50: #choose threshold based on previous max_current
            pass
        if j>50:
            prev_section_start = (j - 1) * section_size
            prev_section_end = prev_section_start + section_size
            l_prev_section = l[prev_section_start:prev_section_end]
            rms_prev_section = np.sqrt(np.mean(l_prev_section ** 2))
            std_dev_prev_section = np.std(l_prev_section)


            # Adjust threshold based on the RMS of the previous section
            threshold_section = rms_prev_section +  3*std_dev_prev_section

            flagged_indices_section = np.nonzero(l_section > threshold_section)[0]

            if flagged_indices_section.shape[0] >=1:
                trigger_section = flagged_indices_section.min()
            
    

                print(f'Trigger Time: {t_section[trigger_section]} at section {j}')
                print(f'Trigger Current: {c_section[trigger_section]} at section {j}')
    l = np.array(losses)
    # Save losses
    np.save(f'/home/maira/Magnets/preprocess_data/losses_qa/l_{k}.npy', l)
    
