import torch
from torch import nn


class FCNModel(nn.Module):
    def __init__(self, input_dim=3, hidden_layers = [16, 32, 16], output_dim=1):
        super().__init__()
        self.linear_layers = []
        self.linear_layers.append(nn.Linear(input_dim,hidden_layers[0]))
        for i in range(len(hidden_layers)-1):
            self.linear_layers.append(nn.Linear(hidden_layers[i],hidden_layers[i+1]))
            self.linear_layers.append(nn.ReLU())
  
        self.linear_layers.append(nn.Linear(hidden_layers[-1],output_dim))

        self.model = nn.Sequential(*self.linear_layers)

    def forward(self,x):

        return self.model(x)
    

class LSTMModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm1 = nn.LSTM(3,64)
    self.linear1 = nn.Linear(64,1)

  def forward(self,x):
    x,(hn,hc) = self.lstm1(x)
    x = self.linear1(x)

    return x