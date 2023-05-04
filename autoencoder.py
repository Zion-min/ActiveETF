
import time
import numpy as np
import pandas as pd
import copy
import pickle
import matplotlib.pyplot as plt
from scipy import optimize
import time
plt.style.use('seaborn')


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class rDAE(nn.Module):
    def __init__(self, n):
        super(rDAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n, int(n*0.5)),
            nn.ReLU(),
            nn.Linear(int(n*0.5), int(n*0.25))
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(n*0.25), int(n*0.5)),
            nn.ReLU(),
            nn.Linear(int(n*0.5), n)

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoEncoder:
    def __init__(self, learning_rate, epoch):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def shrink_universe(self, data, top_rate=0.45, bot_rate=0.05, verbose=False):
        tickers = data.columns
        data_df = copy.deepcopy(data)
        data_with_noise = data + np.random.normal(0,0.001, len(tickers))
        
        #on gpu memory
        data = torch.tensor(data_df.values, device=device).float()
        data_with_noise = torch.tensor(data_with_noise.values).to(device).float()
        #return data, data_with_noise
        model = rDAE(len(tickers)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate)
        loss_func = nn.L1Loss()
        #return model, data_with_noise, data_df
        for e in range(self.epoch):#############
            for i in range(len(data)):
                optimizer.zero_grad()
                output = model.forward(data_with_noise[i])
                loss = loss_func(output[1], data[i])
                loss.backward()
                optimizer.step()
            if e % 10 == 0 and verbose:
                print(f"Epoch {e+1} - MAE Loss : {float(loss)*100}%")
        #학습완료
        predicted = model(data[i])
        res = abs(predicted[1].cpu().detach() - data_df.iloc[i])
        #print("DAE MAE loss on predict :",res.mean(),"%")
        res = res.sort_values().index
        top = res[:int(len(res) * top_rate)]
        bot = res[-int(len(res) * bot_rate):]
        return np.union1d(top, bot)
