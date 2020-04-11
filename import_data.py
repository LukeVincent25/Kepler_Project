import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
import csv
import astroquery
import requests


def import_data():
    base_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?"
    table_name = "cumulative"
    table_url = "table=" + table_name
    format_url = "&format=csv"
    url = base_url + table_url + format_url
    response = requests.get(url)
    data = response.text
    f = open("data.txt", "w+")
    f.write(data)
    f.close
    return


cumulative_data = pd.read_csv('cumulative.csv', sep=" ", delimiter=',')
learning_data_df = pd.read_csv('cumulative.csv', sep=" ", delimiter=',', usecols=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                                                  17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                                  27, 28, 29, 32, 33, 34, 35, 36, 38,
                                                                                  39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                                                                  49))
target = cumulative_data['koi_score']
tensor_output = torch.tensor(target.values.astype(np.float64))
tensor_tmp = learning_data_df.values.astype(np.float64)
tensor_input = torch.from_numpy(tensor_tmp)


class Net(nn.Module):
    def __init__(self):

        D_in = 40  # D_in dimensions in
        H = 20      # H hidden dimensions
        D_out = 1   # D_out dimensions out

        super(Net, self).__init__()
        self.fc1 = nn.Linear(D_in, H, 1)
        self.fc2 = nn.Linear(H, D_out, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class PytorchData(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __getitem__(self, index):
        return self.input[index, :], self.output[index]

    def __len__(self):
        return len(self.output)


NN = Net()
optimizer = optim.SGD(NN.parameters(), lr=0.01, momentum=0.9)

# DataSet
dataset = PytorchData(tensor_input, tensor_output)

# Dataloader
N = 1 # batch size
dataloader = DataLoader(dataset, N, shuffle=True)


i = 0
for [data_input, target] in dataloader:
    optimizer.zero_grad()
    output = NN(data_input)
    # criterion = nn.MSELoss()
    # loss = criterion(output, target[i])
    # loss.backward()
    # optimizer.step()
    # i += 1

    print(data_input.shape)
