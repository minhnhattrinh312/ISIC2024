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
df_train_2019 = df_train_2019[df_train_2019["target"] == 1]
# loop through the dataframe and add the address of the image to the dataframe
for isic_id in tqdm(df_train_2019["isic_id"]):
    image_path1 = f"./dataset/data2019/image/{isic_id}.jpg"
    image_path2 = f"./dataset/data2019/image/{isic_id}_downsampled.jpg"

    if os.path.exists(image_path1):
        df_train_2019.loc[df_train_2019["isic_id"] == isic_id, "image_path"] = image_path1
    elif os.path.exists(image_path2):
        df_train_2019.loc[df_train_2019["isic_id"] == isic_id, "image_path"] = image_path2
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
df_train_2020 = df_train_2020[df_train_2020["target"] == 1]

for isic_id in tqdm(df_train_2020["isic_id"]):
    image_path = f"./dataset/data2020/image/{isic_id}.jpg"

    if os.path.exists(image_path):
        df_train_2020.loc[df_train_2020["isic_id"] == isic_id, "image_path"] = image_path
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
df_train_2024 = pd.concat(
    [df_train_2024[df_train_2024["target"] == 1], df_train_2024[df_train_2024["target"] == 0].head(50000)]
)
# add image_path to the dataframe
for isic_id in tqdm(df_train_2024["isic_id"]):
    image_path = f"./dataset/data2024/image/{isic_id}.jpg"
    if os.path.exists(image_path):
        df_train_2024.loc[df_train_2024["isic_id"] == isic_id, "image_path"] = image_path
    else:
        print(f"Image {isic_id} not found")
        break

# merge all dataframes
df_train = pd.concat([df_train_2019, df_train_2020, df_train_2024], axis=0)
# keep only the columns: isic_id, target, image_path
df_train = df_train[["isic_id", "target", "image_path"]]
# change index of the dataframe from 0 to len(df)
df_train.reset_index(drop=True, inplace=True)
# create a new column for the dataset which is fold, that split the dataset into 5 folds
# each fold has 20% of the dataset and the target == 1 will be balanced in each fold

df_train["fold"] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(skf.split(df_train, df_train["target"])):
    df_train.loc[test_index, "fold"] = fold + 1

# save the dataframe for 55000 images
df_train.to_csv("./dataset/data_images.csv", index=False)
