import pandas as pd
from pathlib import Path
import torch
from sklearn.preprocessing import MinMaxScaler

class PureTestDataset:
    def __init__(self, data, labels, device):
        self.data = data
        self.labels = labels
        self.device = device

    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.float).to(self.device).view(-1,1), torch.tensor(self.labels[i], dtype=torch.float).to(self.device)

    def __len__(self):
        return len(self.data)

class PureTrainDataset:
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.float).to(self.device).view(-1,1)

    def __len__(self):
        return len(self.data)

class HandCraft:
    def __init__(self, filepath):
        self.filepath = filepath
        self.load()

    def load(self):
        df = pd.read_csv(self.filepath, names=["data", "labels"], header=None, index_col=False)
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(df["data"].values.reshape(-1,1)).reshape(-1)
        self.data = df["data"].values
        self.labels = df["labels"].values

    def slidding_window_on_data(self, data, window_size, step, drop_last=False):
        slidded_data = []
        start = 0
        end = start + window_size
        while end <= len(data):
            slidded_data.append(data[start:end])
            start += step
            end = start + window_size
        return slidded_data

    def slidding_window_on_labels(self, labels, window_size, step, drop_last=False):
        slidded_labels = []
        slidded_window_labels = self.slidding_window_on_data(labels, window_size, step, drop_last)
        for slidded_window_label in slidded_window_labels:
            slidded_labels.append(slidded_window_label[-1])
        return slidded_labels

if __name__ == "__main__":

    filepath = Path("../AD-Plot/data/spike.csv")
    window_size = 100
    step = 1
    drop_last = False

    hc_spike = HandCraft(filepath)
    train_len = int(len(hc_spike.data)/2)
    train_data, test_data = hc_spike.data[:train_len], hc_spike.data[train_len:]
    import pdb; pdb.set_trace()
    train_labels, test_labels = hc_spike.labels[:train_len], hc_spike.labels[train_len:]
    train_data = hc_spike.slidding_window_on_data(train_data, window_size, step, drop_last)
    test_data = hc_spike.slidding_window_on_data(test_data, window_size, step, drop_last)
    test_labels = hc_spike.slidding_window_on_labels(test_labels, window_size, step, drop_last)

    device = torch.device("cuda:1")
    train_dataset = PureTrainDataset(train_data, device)
    test_dataset = PureTestDataset(test_data, test_labels, device)

    import pdb; pdb.set_trace()
