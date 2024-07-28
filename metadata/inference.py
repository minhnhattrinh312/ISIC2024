from tqdm import tqdm
import torch
from metrics import compute_r2_score, compute_rmse, compute_mse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import TempData, TempDataSequence
import matplotlib.pyplot as plt

def infer_per_sample(sample,config,model,y_scaler):
    X = sample['x']
    y = sample['y']

 #   X, y = X.to(config['device']), y.to(config['device'])

    y_pred = model(X)

    y_pred = torch.from_numpy(y_scaler.inverse_transform(y_pred.numpy()))
    y = torch.from_numpy(y_scaler.inverse_transform(y))

    mse = compute_mse(y, y_pred)
    rmse = compute_rmse(y, y_pred)
    r2_score = compute_r2_score(y,y_pred)

    
    return rmse, mse, r2_score

def infer_per_sample_sequence(sample,config,model,y_scaler):
    X = sample['u']
    x0 = sample['x0']
    y = sample['y']
    t = sample['time']

    y_pred_list = []
    for i in range(config['horizon']):
      y_pred_list.append(model(X,x0,t[:,i:i+1]))

    y_pred = torch.concatenate(y_pred_list, axis = 1)

    y_pred = torch.from_numpy(y_scaler.inverse_transform(y_pred.numpy()))
    y = torch.from_numpy(y_scaler.inverse_transform(y))

    mse = compute_mse(y, y_pred)
    rmse = compute_rmse(y, y_pred)
    r2_score = compute_r2_score(y,y_pred)

    
    return rmse, mse, r2_score

def infer_data(config, model, valid_loader, y_scaler):
    model.eval()
    r2_score_list = []
    rmse_loss_list = []
    mse_loss_list = []

    with torch.no_grad():
        for sample in tqdm(valid_loader, desc="Evaluating", leave=False):
            rmse_loss, mse_loss, r2_score = \
            infer_per_sample(sample, config, model, y_scaler) if not config['sequence'] \
            else infer_per_sample_sequence(sample, config, model, y_scaler) 
            r2_score_list.append(r2_score.cpu().numpy())
            mse_loss_list.append(mse_loss.cpu().numpy())
            rmse_loss_list.append(rmse_loss.cpu().numpy())

    mse = np.mean(mse_loss_list)
    rmse = np.mean(rmse_loss_list)
    r2_score = np.mean(r2_score_list)
    print(f"\t R2 score: {r2_score:.5f} |  MSE Loss: {mse:.8f} |  RMSE loss: {rmse:.8f}")
    return  rmse, mse, r2_score

def prepare_test_loader(config, data):
  if not config['sequence']:
    test_data = TempData(data['x_test'],data['y_test'])
  else:
    test_data = TempDataSequence(data['x_test'],data['y_test'],config,None)

  test_dataloader = DataLoader(test_data, batch_size = config['batch_size'], shuffle = True)

  return test_dataloader

def infer_numpy_data_one(data, model, x_scaler, y_scaler):
    x_test = torch.from_numpy(data['x_test']).to(dtype=torch.float32)
    y_test = data['y_test']

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        
    y_pred = torch.from_numpy(y_scaler.inverse_transform(y_pred.numpy()))
    y_test = torch.from_numpy(y_scaler.inverse_transform(y_test))

    mse = compute_mse(y_test, y_pred).numpy()
    rmse = compute_rmse(y_test, y_pred).numpy()
    r2_score = compute_r2_score(y_test, y_pred).numpy()

    print(f"\t R2 score: {r2_score:.5f} |  MSE Loss: {mse:.8f} |  RMSE loss: {rmse:.8f}")

    return y_test.numpy(), y_pred.numpy()

def infer_numpy_data_sequence(data, model, x_scaler, y_scaler, config):
    x_test = torch.from_numpy(data['x_test']).to(dtype=torch.float32)
    y_test = data['y_test'][:,:,0]

    x0 = x_test[:,0:1,0]
    u = x_test[:,:,1:]
    t = torch.linspace(1,config['horizon'],config['horizon']).repeat(len(u),1)

    y_pred_list = []
    model.eval()
    model.to(device = 'cpu')
    with torch.no_grad():
      for i in range(config['horizon']):
        y_pred_list.append(model(u,x0,t[:,i:i+1]))

    y_pred = torch.concatenate(y_pred_list, axis = 1)

    y_pred = y_scaler.inverse_transform(y_pred.numpy())
    y_test = y_scaler.inverse_transform(y_test)

    mse ,rmse, r2_score = compute_metrics(y_test,y_pred)

    print(f"\t R2 score: {r2_score:.5f} |  MSE Loss: {mse:.8f} |  RMSE loss: {rmse:.8f}")

    return y_test, y_pred

def compute_metrics(y_test,y_pred):
    y_pred = torch.from_numpy(y_pred)
    y_test = torch.from_numpy(y_test)
    mse = compute_mse(y_test, y_pred).numpy()
    rmse = compute_rmse(y_test, y_pred).numpy()
    r2_score = compute_r2_score(y_test, y_pred).numpy()

    return mse, rmse, r2_score


def infer_numpy_data(data, model, x_scaler, y_scaler, config):
  if config['sequence']:
    return infer_numpy_data_sequence(data, model, x_scaler, y_scaler, config)
  else:
    return infer_numpy_data_one(data, model, x_scaler, y_scaler)


def plot_sequence(y_true,y_pred, n, config, f_size = (10,10)):
  num_channels = config['horizon']
  num_cols = 2  # number of columns in the grid

  num_rows = num_channels // num_cols
  if num_channels % num_cols != 0:
    num_rows += 1

  fig, axs = plt.subplots(num_rows, num_cols, figsize= f_size)

  for i in range(num_channels):
    mse, rmse, r2_score = compute_metrics(y_true[:,i],y_pred[:,i])
    row_i = i // num_cols
    col_i = i % num_cols
    axs[row_i, col_i].plot(y_true[:n, i], label='Y true')
    axs[row_i, col_i].plot(y_pred[:n, i], label='Y pred')
    axs[row_i, col_i].legend()
    axs[row_i, col_i].set_xlabel('Temperature')
    axs[row_i, col_i].set_ylabel('Samples')
    axs[row_i, col_i].set_title(f'Time step {i} \n R2 score = {rmse:.3f} || RMSE = {rmse:.3f} || MSE = {mse:3f}')
    
  plt.tight_layout()
  plt.show()

def plot_one(y_true,y_pred,n, f_size = (16,5)):
  plt.figure(figsize=f_size)
  plt.plot(y_pred[:n], label='Prediction')
  plt.plot(y_true[:n], label='Ground Truth')
  plt.xlabel('t')
  plt.ylabel('Temperature')
  plt.legend()
  plt.show()

def plot_data(y_true,y_pred, n, config, f_size = (10,10)):
  if config['sequence']:
    plot_sequence(y_true,y_pred, n, config,f_size)
  else:
    plot_one(y_true,y_pred,n,f_size)


