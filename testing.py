import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pickle as pkl
import os
import numpy as np
from pprint import pprint
import json
from eval_methods import bf_search


def cal_performance(input, reconstruct_input):
    return F.mse_loss(input, reconstruct_input, reduction="none")


def test(model, test_dataloader, config):
    ''' Start training '''
    model.eval()

    start = time.time()
    test_score, test_label = get_scores(model, test_dataloader, config)
    best_valid_metrics = get_metrics(test_score, test_label, config.bf_search_min, config.bf_search_max, config.bf_search_step_size, config.display_freq)
    save_results(best_valid_metrics, config.test_score_label_save)

    print("testing time cost: ", time.time()-start)
    print('=' * 30 + 'result' + '=' * 30)
    pprint(best_valid_metrics)

def save_results(best_valid_metrics, test_score_label_save):

    result_file = os.path.join(test_score_label_save, "result.json")
    with open(result_file, "w") as f:
        json.dump(best_valid_metrics, f, indent=4)


def get_scores(model, test_dataloader, config):

    desc = '  - (testing)   '
    score_list = []
    label_list = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, mininterval=2, desc=desc, leave=False):
            # prepare data
            src_seq, labels = batch

            # forward
            reconstruct_input = model(src_seq)

            reconstruct_diff = cal_performance(
                input=src_seq,
                reconstruct_input=reconstruct_input
            )
            # 打分的评价：只取当前窗口的最后一个的重构误差当作是本窗口的误差
            score_list.extend(reconstruct_diff.sum(-1)[:,-1].detach().cpu().numpy().tolist())
            label_list.extend(labels.detach().cpu().numpy().tolist())

    # 需要把score_list以及label_list存下来，以便于之后的计算
    if not os.path.exists(config.test_score_label_save):
        os.makedirs(config.test_score_label_save)
    with open(os.path.join(config.test_score_label_save, "score_label.pkl"), "wb") as f:
        pkl.dump([score_list, label_list], f)

    test_score = np.array(score_list, dtype=float)
    test_label = np.array(label_list, dtype=float)

    return test_score, test_label

def get_metrics(test_score, test_label, bf_search_min, bf_search_max, bf_search_step_size, display_freq):

    t, th = bf_search(test_score,
                      test_label,
                      start=bf_search_min,
                      end=bf_search_max,
                      step_num=int(abs(bf_search_max - bf_search_min) /
                                   bf_search_step_size),
                      display_freq=display_freq)

    best_valid_metrics = {}
    best_valid_metrics.update({
        'best-f1': float(t[0]),
        'precision': float(t[1]),
        'recall': float(t[2]),
        'TP': int(t[3]),
        'TN': int(t[4]),
        'FP': int(t[5]),
        'FN': int(t[6]),
        'latency': float(t[-1]),
        'threshold': float(th)
    })
    # best_valid_metrics.update(pot_result)
    return best_valid_metrics
