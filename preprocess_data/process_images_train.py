import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# add path to this file
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# create a df for 3 datasets but get the positive samples from 2019 and 2020 only
# load the data
df_train_2019a = pd.read_csv("./dataset/data2019/ISIC_2019_Training_Metadata.csv")
df_train_2019b = pd.read_csv("./dataset/data2019/train-groundtruth.csv")
# concatenate the two dataframes, cat 2019a with 2019b
df_train_2019 = pd.concat([df_train_2019a, df_train_2019b], axis=1)
# remove columns :lesion_id,Unnamed: 0
df_train_2019 = df_train_2019.drop(columns=["Unnamed: 0", "isic_id", "lesion_id"], axis=1)
df_train_2019 = df_train_2019.rename(columns={"image": "isic_id"})

# get the positive samples from 2019
df_train_2019_positives = df_train_2019[df_train_2019["target"] == 1].copy()
# loop through the dataframe and add the address of the image to the dataframe
for isic_id in tqdm(df_train_2019_positives["isic_id"]):
    image_path1 = f"./dataset/data2019/image/{isic_id}.jpg"
    image_path2 = f"./dataset/data2019/image/{isic_id}_downsampled.jpg"

    if os.path.exists(image_path1):
        df_train_2019_positives.loc[df_train_2019_positives["isic_id"] == isic_id, "image_path"] = image_path1
    elif os.path.exists(image_path2):
        df_train_2019_positives.loc[df_train_2019_positives["isic_id"] == isic_id, "image_path"] = image_path2
    else:
        print(f"Image {isic_id} not found")
        break

df_train_2020 = pd.read_csv("./dataset/data2020/train.csv")
# drop the diagnosis, benign_malignant columns
df_train_2020 = df_train_2020.drop(columns=["diagnosis", "benign_malignant"], axis=1)
# change name of the columns 2020 to the same 2019:
# image_name -> image, anatom_site_general_challenge -> anatom_site_general
df_train_2020 = df_train_2020.rename(
    columns={"image_name": "isic_id", "anatom_site_general_challenge": "anatom_site_general"}
)

# get the positive samples from 2020
df_train_2020_positives = df_train_2020[df_train_2020["target"] == 1].copy()

for isic_id in tqdm(df_train_2020_positives["isic_id"]):
    image_path = f"./dataset/data2020/image/{isic_id}.jpg"

    if os.path.exists(image_path):
        df_train_2020_positives.loc[df_train_2020_positives["isic_id"] == isic_id, "image_path"] = image_path
    else:
        print(f"Image {isic_id} not found")
        break

# load data 2024
df_train_2024 = pd.read_csv("./dataset/data2024/train-metadata.csv")
df_test_2024 = pd.read_csv("./dataset/data2024/test-metadata.csv")
# remove columns in df 2024 train if it not in df 2024 test
remove_columns = [col for col in df_train_2024.columns if col not in df_test_2024.columns]
remove_columns.remove("target")
df_train_2024 = df_train_2024.drop(columns=remove_columns, axis=1)
# get all images with target == 1 and 50000 images with target == 0
# to balance the dataset
df_train_2024 = df_train_2024.sort_values(by="target", ascending=False)

df_train_2024_positives = df_train_2024[df_train_2024["target"] == 1].copy()
for isic_id in tqdm(df_train_2024_positives["isic_id"]):
    image_path = f"./dataset/data2024/image/{isic_id}.jpg"

    if os.path.exists(image_path):
        df_train_2024_positives.loc[df_train_2024_positives["isic_id"] == isic_id, "image_path"] = image_path
    else:
        print(f"Image {isic_id} not found")
        break

df_train_positives = pd.concat([df_train_2019_positives, df_train_2020_positives, df_train_2024_positives], axis=0)
df_train_positives = df_train_positives.reset_index(drop=True)
# spilt df_train_positives into 5 folds, each fold has the same number of positive samples
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df_train_positives = df_train_positives[["isic_id", "target", "image_path"]]
df_train_positives["fold"] = -1
for fold, (train_index, val_index) in enumerate(skf.split(df_train_positives, df_train_positives["target"])):
    df_train_positives.loc[val_index, "fold"] = fold + 1

# get the negative samples from 2024
df_train_2024 = df_train_2024[["isic_id", "target"]]
df_train_2024_negatives = df_train_2024[df_train_2024["target"] == 0].copy()
# for training each fold, we use 44000 negative samples for train and 5500 negative samples for validation from df_train_2024_negatives
for fold in range(1, 6):
    df_negatives = df_train_2024_negatives[55000 * (fold - 1) : 55000 * fold].copy().reset_index(drop=True)
    # get 11000 negative samples for validation so change the "fold" column to -1
    df_negatives.loc[44000:, "fold"] = fold
    df_negatives.loc[:44000, "fold"] = -1
    for isic_id in tqdm(df_negatives["isic_id"]):
        image_path = f"./dataset/data2024/image/{isic_id}.jpg"

        if os.path.exists(image_path):
            df_negatives.loc[df_negatives["isic_id"] == isic_id, "image_path"] = image_path
        else:
            print(f"Image {isic_id} not found")
            break
    # concatenate the negative samples with the positive samples
    df_train_fold = pd.concat([df_negatives, df_train_positives], axis=0).reset_index(drop=True)
    # save the dataframe to csv file
    df_train_fold.to_csv(f"./dataset/data_images_fold{fold}.csv", index=False)
