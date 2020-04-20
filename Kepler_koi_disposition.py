import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = ROOT_DIR + "\\models"


# function to call api command to retrieve koi cumulative table
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
learning_data_df = pd.read_csv('cumulative.csv', sep=" ", delimiter=',', usecols=(4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                                                  17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                                  27, 28, 29, 32, 33, 34, 35, 36, 38,
                                                                                  39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                                                                  49))
# drop all rows with nan's
trimmed_df = learning_data_df.dropna()

# remove any undetermined exo-planets "Candidates"
trimmed_clean_df = trimmed_df[trimmed_df.koi_disposition != "CANDIDATE"]

# set koi score as targets
target = trimmed_clean_df['koi_score']

# convert labels to 1's and 0's
labels = trimmed_clean_df['koi_disposition']
labels.loc[labels == 'CONFIRMED'] = 1
labels.loc[labels == 'FALSE POSITIVE'] = 0


# convert to tensor
tensor_output = torch.tensor(target.values.astype(np.float))
label_output = torch.tensor(labels.values.astype(np.float))

# remove columns for data crunching
input_data = trimmed_clean_df.drop(columns=['koi_score', 'koi_disposition'])

# convert to tensor
tensor_tmp = input_data.values.astype(np.float)
tensor_input = torch.from_numpy(tensor_tmp)


class Net(nn.Module):
    def __init__(self):
        D_in = 40  # D_in dimensions in
        H1 = 100      # H hidden dimensions
        H2 = 50
        D_out = 1   # D_out dimensions out

        super(Net, self).__init__()
        self.fc1 = nn.Linear(D_in, H1, 1)
        self.fc2 = nn.Linear(H1, H2, 1)
        self.fc3 = nn.Linear(H2, D_out, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class PytorchData(Dataset):
    def __init__(self, input, output):
        self.cols = len(input[0, :])
        self.rows = len(input[:, 0])
        self.input = self.normalize_data(input.float())
        self.output = output.float()

    def __getitem__(self, index):
        return self.input[index, :], self.output[index]

    def __len__(self):
        return len(self.output)

    def normalize_data(self, x):
        for c in range(self.cols):
            max_value_col = torch.max(x[:, c])
            if max_value_col != 0:
                x[:, c] = (max_value_col - x[:, c]) / max_value_col
        return x


NN = Net()
learning_rate = .00005
optimizer = optim.Adam(NN.parameters(), lr=learning_rate)

# DataSet
# koi_score nn
# dataset = PytorchData(tensor_input, tensor_output)

# koi_disposition
dataset = PytorchData(tensor_input, label_output)

# Dataloader
N = 10  # batch size
dataloader = DataLoader(dataset, N, shuffle=True)


criterion = nn.MSELoss(reduction='sum')
iterations = 50
plot_loss = np.zeros(iterations)


# CHECK IF LOADING OR CREATING A NN MODEL
PATH = 'mode.pth'
loading_model = True
if loading_model:
    NN.load_state_dict(torch.load(PATH))
    NN.eval()

# CHECK IF LEARNING
learning = False
if learning:
    for k in range(iterations):
        total_loss = 0
        for [x_in, y] in dataloader:
            for data_input, target in zip(x_in, y):
                output = NN(data_input)
                loss = criterion(output, target)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        plot_loss[k] = total_loss
        print(k)


torch.save(NN.state_dict(), PATH)

# CHECK IF TESTING MODEL FOR ACCURACY
testing = True
if testing:
    print("testing accuracy")
    N = 1
    dataloader = DataLoader(dataset, N, shuffle=True)

    count = 0
    correct = 0
    for [data_input, target] in dataloader:
        count += 1
        output = NN(data_input)


        if output > .5:
            prediction = 1
        else:
            prediction = 0

        if prediction == target:
            correct += 1
        else:
            # check values
            print("predicted: " + str(output))
            print("actual: " + str(target))

    acc = correct / count
    print("Accuracy: " + str(acc))




plt.figure(1)
t = range(iterations)
plt.plot(t, plot_loss)
plt.xlabel('t')
plt.ylabel('loss')
plt.title('loss vs iterations')
plt.show()

