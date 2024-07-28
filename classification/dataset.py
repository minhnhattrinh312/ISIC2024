from torch.utils.data import Dataset
import torch
from torchvision import transforms
import kornia.augmentation as K
from PIL import Image


# the dataframe will be csv file
class ISIC_Loader(Dataset):
    def __init__(self, csv, mode="train", use_meta=False):
        self.use_meta = use_meta
        self.mode = mode
        self.image_paths = csv["image_path"].values
        self.targets = csv["target"].values

    def __len__(self):
        # return the number of samples
        return len(self.targets)

    def __getitem__(self, idx):
        # load the image and labels from the csv file
        image = Image.open(self.image_paths[idx]).resize((224, 224))
        label = torch.tensor([self.targets[idx]], dtype=torch.int64)
        image = transforms.ToTensor()(image)
        # image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        if self.use_meta:
            # todo add meta data from csv
            pass
        else:
            data = image

        if self.mode == "train":
            return data, label
        else:
            return data


def get_transform():
    transform = K.container.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.ColorJitter(brightness=0.2, contrast=0.2, p=0.75),
        K.container.AugmentationSequential(
            K.RandomGaussianBlur((5, 5), (0.1, 2), p=0.2),
            K.RandomGaussianNoise(mean=0.0, std=0.02, p=0.2),
            random_apply=(1, 1),
        ),
        K.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), p=0.85),
        K.RandomErasing(scale=(0, 0.2), ratio=(1, 1), value=0.0, p=0.7),
    )
    return transform
