import json
import os
from dataclasses import dataclass
from transformers import HfArgumentParser
import numpy as np
from pprint import pprint


@dataclass
class Config:
    root_dir: str = "./results/MSL_64/"
    filename: str = "result.json"
    metrics: str = "best-f1:latency:precision:recall"


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)

if __name__=="__main__":

    # 读取参数
    parser = HfArgumentParser((Config))
    config = parser.parse_args_into_dataclasses()[0]

    sub_dir = os.listdir(config.root_dir)
    print("total files: {}".format(len(sub_dir)))

    metrics_list_str = config.metrics.strip().split(":")
    metrics_list = [[] for _ in range(len(metrics_list_str))]
    TP = 0
    FN = 0
    FP = 0
    for dir in sub_dir:
        result_file = os.path.join(config.root_dir, dir, config.filename)
        result_dict = load_json(result_file)

        for i, metric in enumerate(metrics_list_str):
            metrics_list[i].append(result_dict[metric])

        TP += result_dict["TP"]
        FN += result_dict["FN"]
        FP += result_dict["FP"]

    metrics_np = np.array(metrics_list)
    metrics_np_mean = metrics_np.mean(-1)

    result_dict = dict()
    for i, dir in enumerate(sub_dir):
        result_dict[dir] = []
        for j in range(len(metrics_list_str)):
            result_dict[dir].append(metrics_list[j][i])
    pprint(result_dict)

    for metrics_str, value in zip(metrics_list_str, metrics_np_mean):
        print(metrics_str, " : ", value)

    print("-"*40)
    recall = TP / (TP+FN)
    precision = TP / (TP+FP)
    f1 = recall*precision*2/(recall + precision)
    print("recall = ", recall)
    print("precision = ", precision)
    print("f1 = ", f1)
