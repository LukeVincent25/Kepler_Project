import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt


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
        D_out = 1   # D_out dimensions out

        super(Net, self).__init__()
        self.fc1 = nn.Linear(D_in, D_out, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
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


# Initiate NN Module and select directory to save
NN = Net()
MODEL_DIR = "models/"
MODEL_NAME = "koi_logisticregression_SGD"
extension = ".pth"



# DataSet
# koi_score nn
# dataset = PytorchData(tensor_input, tensor_output)

# koi_disposition
dataset = PytorchData(tensor_input, label_output)
train_split = .8
train_size = int(train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


# set directory for saving model

PATH = MODEL_DIR + MODEL_NAME + extension

# CHECK IF LOADING OR CREATING A NN MODEL
loading_model = False
if loading_model:
    NN.load_state_dict(torch.load(PATH))
    NN.eval()

# CHECK IF LEARNING
learning = True
if learning:
    # Dataloader
    N = 1  # batch size
    train_loader = DataLoader(train_dataset, N, shuffle=True)

    # Loss calculation method
    criterion = nn.MSELoss(reduction='sum')

    # Initiate Optimizer
    learning_rate = .00001
    optimizer = optim.Adam(NN.parameters(), lr=learning_rate)

    # Iterations
    iterations = 250
    plot_loss = np.zeros(iterations)

    for k in range(iterations):
        total_loss = 0
        for [x_in, y] in train_loader:
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
    test_loader = DataLoader(test_dataset, N, shuffle=True)

    count = 0
    correct = 0
    for [data_input, target] in test_loader:
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

