import torch
from metrics import binary_cross_entropy
from tqdm import tqdm
import numpy as np
from evaluate import evaluate_per_epoch
from utils.train_utils import epoch_time
import time

def train_per_sample(sample, config, model, loss, optimizer):
    X = sample['input']
    y = sample['target']

    X, y = X.to(config['device']), y.reshape((-1,1)).to(config['device'])
    optimizer.zero_grad()
    y_pred = model(X)
    train_loss = loss(y,y_pred)

    train_loss.backward()
    optimizer.step()

    bce = binary_cross_entropy(y, y_pred)

    return train_loss, bce

def train_per_epoch(config, train_dataloader, model, loss, optimizer):
    model.train()
    epoch_loss= 0.0
    bce_list = []

    for sample in tqdm(train_dataloader, desc='training', leave=False):

        train_loss, bce = train_per_sample(sample, config, model, loss, optimizer) 

        epoch_loss += train_loss.item()

        bce_list.append(bce.detach().cpu().numpy())   

    return epoch_loss / len(train_dataloader), np.mean(bce_list)


def train_and_evaluate(config, train_dataloader, valid_dataloader, model, loss, optimizer, scheduler, start_epoch):
    best_value_loss = float('inf')
    start_time = time.monotonic()
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(start_epoch,start_epoch+ config['num_epochs']):
        train_loss, train_bce_score = train_per_epoch(config,train_dataloader,model,loss,optimizer)
        valid_loss, valid_bce_score = evaluate_per_epoch(config,valid_dataloader,model,loss)

        scheduler.step(valid_loss)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print("----------------------------------------------------------------------------------------------")
        print(f"Epoch: {epoch+1} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print("Training")
        print(
        f"\t BCE Loss: {train_loss:.8f} | bce score: {train_bce_score:.5f}")
        print("Validating")
        print(
        f"\t BCE Loss: {valid_loss:.8f} |  bce score: {valid_bce_score:.5f}")
        print("----------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------")

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

    return train_loss_list, valid_loss_list






