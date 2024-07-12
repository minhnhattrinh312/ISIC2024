from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from PIL import Image
import numpy as np
from skimage import exposure
from classification import config


# todo: check the csv file
class ClsLoader(Dataset):
    def __init__(self, path_image, csv, transform=True, mode="train", use_meta=False):
        self.path_image = path_image
        self.csv = csv
        self.use_meta = use_meta
        self.mode = mode
    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):

        image = Image.open(f"{self.path_image}{self.ids[idx]}.png")
        label = torch.tensor(self.csv.iloc[index][self.meta_features]).float()

        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == "test":
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()
