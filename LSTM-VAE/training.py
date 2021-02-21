from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
import os
from testing import get_scores, get_metrics
from pprint import pprint

def test_epoch(model, test_dataloader, config):
    model.eval()

    start = time.time()
    test_score, test_label = get_scores(model, test_dataloader, config)
    best_valid_metrics = get_metrics(test_score, test_label, config.bf_search_min, config.bf_search_max, config.bf_search_step_size, config.display_freq)

    print("testing time cost: ", time.time()-start)
    print('=' * 30 + 'result' + '=' * 30)
    pprint(best_valid_metrics)

def eval_epoch(model, eval_dataloader):

    model.eval()
    desc = '  - (validation)   '
    total_loss = 0.0
    batch_num = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, mininterval=2, desc=desc, leave=False):
            # prepare data
            src_seq = batch

            # forward
            reconstruct_input = model(src_seq)

            loss = cal_performance(
                input=src_seq,
                reconstruct_input=reconstruct_input
            )
            # note keeping
            total_loss += loss.item()
            batch_num += 1

    return total_loss/batch_num

def cal_performance(input, reconstruct_input):
    return F.mse_loss(input, reconstruct_input)

def train_epoch(model, train_dataloader, optimizer):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0.0
    batch_num = 0

    desc = '  - (Training)   '
    for batch in tqdm(train_dataloader, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = batch

        # forward
        optimizer.zero_grad()
        reconstruct_input = model(src_seq)

        # backward and update parameters
        loss = cal_performance(
            input=src_seq,
            reconstruct_input=reconstruct_input
        )
        loss.backward()
        optimizer.step()

        # note keeping
        total_loss += loss.item()
        batch_num += 1

    return total_loss/batch_num

def train_epoch_VSAE(model, train_dataloader, optimizer, KL_weight=1.):
    ''' Epoch operation in training phase'''

    model.train()
    rec_total_loss = 0.0
    KL_total_loss = 0.0
    batch_num = 0

    desc = '  - (Training)   '
    for batch in tqdm(train_dataloader, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = batch

        # forward
        optimizer.zero_grad()
        mean, log_var = model.encode_and_variational(src_seq)
        z = model.encoder_to_decoder(model.reparameter(mean, log_var))
        reconstruct_input = model.decode(src_seq, z)

        # backward and update parameters
        # reconstruction loss
        # batch_size
        reconstruction_loss = F.mse_loss(src_seq, reconstruct_input, reduction="none").sum(-1).mean(-1)
        # KL loss
        # batch_size
        KL_loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1).sum(-1)

        loss = reconstruction_loss + KL_loss * KL_weight
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        # note keeping
        rec_total_loss += reconstruction_loss.mean().item()
        KL_total_loss += KL_loss.mean().item()
        batch_num += 1

    return rec_total_loss/batch_num, KL_total_loss/batch_num

def train(model, train_dataloader, valid_dataloader, optimizer, config, test_dataloader=None):
    ''' Start training '''

    valid_losses = []
    for epoch_i in range(config.epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss = train_epoch_VSAE(
                model, train_dataloader, optimizer)
        print("train loss ", train_loss, "training time cost: ", time.time()-start)

        start = time.time()
        valid_loss = eval_epoch(model, valid_dataloader)
        print("valid loss ", valid_loss, "validation time cost: ", time.time()-start)

        if test_dataloader is not None:
            test_epoch(model, test_dataloader, config)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': config, 'model': model.state_dict()}

        if not os.path.exists(config.model_save_path):
            os.makedirs(config.model_save_path)

        model_name = config.save_name + '.chkpt'
        model_name = os.path.join(config.model_save_path, model_name)
        if valid_loss <= min(valid_losses):
            torch.save(checkpoint, model_name)
            print('    - [Info] The checkpoint file has been updated.')

