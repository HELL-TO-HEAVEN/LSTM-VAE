from torch.utils.data import DataLoader
import torch
from handcraft_dataset import PureTestDataset, PureTrainDataset, HandCraft


def get_data(datastring, filepath, max_length, step,
             valid_portion=0.3, data_collator=None,
             batch_size=64, device=torch.device("cpu")):

    if datastring == "handcraft":
        #filepath = "../AD-Plot/data/spike.csv"
        #filepath = "../AD-Plot/data/level_shift.csv"
        #filepath = "../AD-Plot/data/pattern_change.csv"
        window_size = max_length
        drop_last = False
        hc_spike = HandCraft(filepath)
        train_len = int(len(hc_spike.data)/2)
        train_data, test_data = hc_spike.data[:train_len], hc_spike.data[train_len:]
        train_labels, test_labels = hc_spike.labels[:train_len], hc_spike.labels[train_len:]
        train_data, val_data = train_data[:int(train_len*(1-valid_portion))], train_data[int(train_len*(1-valid_portion)):]

        train_data = hc_spike.slidding_window_on_data(train_data, window_size, step, drop_last)
        val_data = hc_spike.slidding_window_on_data(val_data, window_size, step, drop_last)
        test_data = hc_spike.slidding_window_on_data(test_data, window_size, step, drop_last)
        test_labels = hc_spike.slidding_window_on_labels(test_labels, window_size, step, drop_last)

        train_set = PureTrainDataset(train_data, device)
        valid_set = PureTrainDataset(val_data, device)
        test_set = PureTestDataset(test_data, test_labels, device)
    else:
        raise NotImplementedError

    return {
        "train_dataloader": DataLoader(dataset=train_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       collate_fn=data_collator),
        "valid_dataloader": DataLoader(dataset=valid_set,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=data_collator),
        "test_dataloader": DataLoader(dataset=test_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=data_collator)
    }
