from transformers import HfArgumentParser, set_seed
import torch
import os

from data import get_data
from models.models import LSTMVAE as Model
from config import Sequence2SequenceConfig
from training import train
from testing import test

def main():

    # 读取参数
    parser = HfArgumentParser((Sequence2SequenceConfig))
    config = parser.parse_args_into_dataclasses()[0]
    print(config)

    # 设置随机种子
    set_seed(config.seed)

    # 定义运行设备
    device = torch.device(config.device)

    # 加载数据集
    data = get_data(
        datastring="handcraft",
        filepath=config.filepath,
        max_length=config.max_length,
        step=config.step,
        valid_portion=config.valid_portion,
        batch_size=config.batch_size,
        device=device
    )

    # 定义模型
    model = Model(
        input_size=config.input_size,
        hidden_size=config.hidden_size
    )
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    test_dataloader = None
    train(model, data["train_dataloader"], data["valid_dataloader"], optimizer, config, test_dataloader=test_dataloader)

    # 模型测试
    # 加载最佳的模型
    model_name = config.save_name + '.chkpt'
    model_name = os.path.join(config.model_save_path, model_name)
    checkpoint = torch.load(model_name)
    print("load best epoch from ", model_name, " best epoch: ", checkpoint["epoch"])
    model.load_state_dict(checkpoint["model"])
    test(model, data["test_dataloader"], config)

if __name__=="__main__":
    main()
