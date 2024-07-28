from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class MelanomaMetaDataset(Dataset):
    def __init__(self, meta_df_x, meta_df_y):

        self.meta_df_x = meta_df_x.reset_index(drop=True)
        self.meta_df_y = meta_df_y.reset_index(drop=True)

    def __len__(self):
        return self.meta_df_x.shape[0]

    def __getitem__(self, index):

        row = self.meta_df_x.iloc[index]
        input = torch.tensor(self.meta_df_x.iloc[index]).float()
        target = torch.tensor(self.meta_df_y.iloc[index]).float()

        sample = {'input': input, 'target': target}
    
        return sample
  