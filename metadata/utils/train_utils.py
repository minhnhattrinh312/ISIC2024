import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from dataset import MelanomaMetaDataset
from networks.fnc_model import *
import scipy.io
from pathlib import Path


def prepare_dataloaders(config, data):

  train_data = MelanomaMetaDataset(data['x_train'],data['y_train'])
  valid_data = MelanomaMetaDataset(data['x_valid'],data['y_valid'])


  train_dataloader = DataLoader(train_data, batch_size = config['batch_size'], shuffle = config['random_split'])
  valid_dataloader = DataLoader(valid_data, batch_size = config['batch_size'], shuffle = config['random_split'])

  return train_dataloader, valid_dataloader

def prepare_objectives(config, model):

    loss = nn.BCEWithLogitsLoss()

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = config['learing_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = config['learning_rate'])
    elif config['optimizer'] == 'Nadam':
        optimizer = optim.NAdam(model.parameters(), lr = config['learning_rate'])

    if config['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

    return loss, optimizer, scheduler

def prepare_models(input_dim,config):
    model = None
    if 'fcn' in config['model']:
        model = FCNModel(input_dim, config['model']['fcn']['hidden_layers'], config['model']['fcn']['output_dim'])
    elif 'lstm' in config['model']:
        model = LSTMModel()
    else:
        raise ValueError("Invalid model configuration")

    return model.to(config['device'])

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_model(config,model):
  print("Saving model state to: ", config['save_model_checkpoint'])
  torch.save({'model_state_dict': model.state_dict(),
            }, config['save_model_checkpoint'])

def save_all_states(config,model,best_valid_loss,optimizer,epoch):
  Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
  print("Saving checkpoints ", config['save_checkpoint'])
  torch.save({'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': best_valid_loss,
              }, config['save_checkpoint'])

    
