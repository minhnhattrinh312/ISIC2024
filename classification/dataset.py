from torch.utils.data import Dataset
import torch
from torchvision import transforms
import kornia.augmentation as K


# the dataframe will be csv file
class ClsLoader(Dataset):
    def __init__(self, csv, mode="train", use_meta=False):
        self.csv = csv
        self.use_meta = use_meta
        self.mode = mode

    def __len__(self):
        # return the number of samples
        return len(self.csv)

    def __getitem__(self, idx):
        # load the image and labels from the csv file
        image = Image.open(self.csv.iloc[idx]["image_path"]).resize((224, 224))
        label = torch.tensor(self.csv.iloc[idx]["target"]).long()

        image = transforms.ToTensor()(image)
        # image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        if self.use_meta:
            # todo add meta data from csv
            pass
            # data = image, torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = image

        if self.mode == "train":
            return data, label
        else:
            return data


def get_transform():
    transforms_train = K.container.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.ColorJitter(brightness=0.2, contrast=0.2, p=0.75),
        K.container.AugmentationSequential(
            K.RandomGaussianBlur((5, 5), (0.1, 2), p=0.5),
            K.RandomGaussianNoise(mean=0.0, std=0.02, p=0.5),
            random_apply=(1, 1),
        ),
        # K.RandomClahe(clip_limit=(2, 2), grid_size=(2, 2), p=0.7),
        K.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 0.9), p=0.85),
        K.RandomErasing(scale=(0, 0.3), ratio=(1, 1), value=0.0, p=0.7),
        K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    )

    transforms_val = K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))

    return transforms_train, transforms_val
