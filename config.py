from dataclasses import dataclass

@dataclass
class Sequence2SequenceConfig:

    # 设置随机种子
    seed: int = 2020

    # 数据集相关参数
    datastring: str = "SMD"
    max_length: int = 100
    step: int = 1
    filepath: str = "../AD-Plot/data/spike.csv"
    #filepath: str = "../AD-Plot/data/pattern_change.csv"
    #filepath: str = "../AD-Plot/data/level_shift.csv"
    labelfilepath: str = ""
    testfilepath: str = ""

    # 网络相关参数
    input_size: int = 1
    hidden_size: int = 16
    dropout: float = 0.2

    # 训练相关参数
    valid_portion: float = 0.3
    shuffle: bool = True
    batch_size: int = 64
    lr: float = 0.01
    epochs: int = 400
    # 是否在训练过程中输出测试结果
    train_test: bool = False

    g_lr: float = 0.01
    d_lr: float = 0.01

    # 运行设备
    device: str = "cuda:2"

    # 结果保存
    save_name: str = "spike"
    model_save_path: str = "./models/"
    test_score_label_save: str = "./results/HandCraft_Spike/"
    #test_score_label_save: str = "./results/HandCraft_LevelShift/"
    #test_score_label_save: str = "./results/HandCraft_PatternChange_predictive/"
    bf_search_min: float = 0
    bf_search_max: float = 30
    bf_search_step_size: float = 0.01
    display_freq: int = 1000

    # 与S-RNNs有关的参数
    # ensemble的数量
    ensemble_space: int = 20
