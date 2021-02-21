import torch
from pathlib import Path
import pickle

from models.models import VSAE as Model
from data import get_data

def get_embedding(model, test_dataloader):

    mean_list = []
    log_var_list = []
    label_list = []
    with torch.no_grad():
        for batch in test_dataloader:
            src_seq, labels = batch
            # mean, log_var: batch_size, 2
            mean, log_var = model.encode_and_variational(src_seq)

            mean_list.extend(mean.detach().cpu().tolist())
            log_var_list.extend(log_var.detach().cpu().tolist())
            label_list.extend(labels.detach().cpu().numpy().tolist())

    return mean_list, log_var_list, label_list

def get_embedding_and_dump(model, test_dataloader, savepath):

    savepath = Path(savepath)
    mean, log_var, label = get_embedding(model, test_dataloader)
    pickle.dump({"mean":mean, "log_var":log_var, "label":label}, savepath.open("wb"))

if __name__ == "__main__":

    device = torch.device("cuda:1")

    # spike
    model_path = "./models/best_spike.chkpt"
    savepath = "./emb/spike.pkl"
    filepath = "../AD-Plot/data/spike.csv"
    # level shift
    #model_path = "./models/best_levelshift.chkpt"
    #savepath = "./emb/level_shift.pkl"
    #filepath = "../AD-Plot/data/level_shift.csv"
    # pattern change
    #model_path = "./models/best_patternchange.chkpt"
    #savepath = "./emb/pattern_change.pkl"
    #filepath = "../AD-Plot/data/pattern_change.csv"

    model = Model(input_size=1, hidden_size=2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    data = get_data(
        datastring="handcraft",
        filepath = filepath,
        max_length=None,
        step=None,
        labelfilepath=None,
        testfilepath=None,
        valid_portion=None,
        shuffle=None,
        batch_size=128,
        device=device
    )

    get_embedding_and_dump(model, data["test_dataloader"], savepath)
