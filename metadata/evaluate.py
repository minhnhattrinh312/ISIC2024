from tqdm import tqdm
import torch
from metrics import binary_cross_entropy
import numpy as np

def evaluate_per_sample(sample,config,model,loss):
    X = sample['input']
    y = sample['target']

    X, y = X.to(config['device']), y.to(config['device']).reshape((-1,1))

    y_pred = model(X)
    evaluate_loss = loss(y,y_pred)

    bce = binary_cross_entropy(y, y_pred)

    return evaluate_loss, bce


def evaluate_per_epoch(config, valid_loader,model, loss):
    model.eval()
    bce_list = []
    epoch_loss = 0.0
    with torch.no_grad():
        for sample in tqdm(valid_loader, desc="Evaluating", leave=False):
            evaluate_loss, bce=  evaluate_per_sample(sample, config, model, loss) 

            epoch_loss += evaluate_loss.item()
            bce_list.append(bce.detach().cpu().numpy())

    return epoch_loss/len(valid_loader), np.mean(bce_list)

