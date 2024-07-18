import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import torch
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, "/ISIC2024/")
from classification import *
from pytorch_lightning.loggers import WandbLogger
import os
import csv
import wandb


subjects_dir = "./data/data_isbi_2015/isbi2npz2D/"
torch.set_float32_matmul_precision("high")


# Main function
if __name__ == "__main__":
    # Loop over the folds
    # for i in range(1, 6):
    i = 1
    cfg.TRAIN.FOLD = i
    print("train on fold", cfg.TRAIN.FOLD)
    cfg.DIRS.SAVE_DIR = f"./weights_isbi2.5D_{cfg.PREDICT.MODEL}_{cfg.TRAIN.LOSS}/fold{cfg.TRAIN.FOLD}/"
    model = convnext_small(
        pretrained=cfg.TRAIN.PRETRAIN,
        in_22k=cfg.TRAIN.CONVEXT.IN22K,
        in_chans=cfg.DATA.IN_CHANNEL,
        num_classes=cfg.DATA.NUM_CLASS,
        drop_path_rate=cfg.TRAIN.CONVEXT.DROPOUT,
    )
    # create folder to save checkpoints
    os.makedirs(cfg.DIRS.SAVE_DIR, exist_ok=True)
    # List of subjects in test set
    list_test_subject = sorted(glob.glob(f"./data/data_isbi_2015/training/*{cfg.TRAIN.FOLD}/preprocessed/*flair*"))

    # List of subjects in the training set
    list_train_subject = []
    segmenter = Segmenter(
        model,
        cfg.DATA.CLASS_WEIGHT,
        cfg.DATA.NUM_CLASS,
        cfg.OPT.LEARNING_RATE,
        cfg.OPT.FACTOR_LR,
        cfg.OPT.PATIENCE_LR,
    )

    for row in rows:
        name_subject = row["name"]
        fold = int(row["fold"])
        if fold != cfg.TRAIN.FOLD:
            list_train_subject.append(f"{subjects_dir}{name_subject}.npz")

    # Create a ISBILoader object for the training data using the list_train_subject variable and the defined augmentations
    train_data = ISBILoader(list_subject=list_train_subject)

    # Create a ISBILoader object for the test data using the list_test_subject variable
    # test_data = ISBILoader(list_subject=list_test_subject)
    test_dataset = ISBI_Test_Loader(list_test_subject)

    # If wandb_logger is True, create a WandbLogger object
    if cfg.TRAIN.WANDB:
        wandb_logger = WandbLogger(
            project="isbi",
            group=f"tiramisu_{cfg.TRAIN.LOSS}",
            name=f"fold{cfg.TRAIN.FOLD}_{cfg.TRAIN.LOSS}",
            resume="allow",
        )
    else:
        wandb_logger = False
    # Define data loaders for the training and test data
    train_dataset = DataLoader(
        train_data,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        drop_last=True,
        prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
    )
    # test_dataset = DataLoader(test_data, batch_size=cfg.TRAIN.BATCH_SIZE,
    #                         num_workers=cfg.TRAIN.NUM_WORKERS, prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR)

    # Initialize a ModelCheckpoint callback to save the model weights after each epoch
    check_point_score = pl.callbacks.model_checkpoint.ModelCheckpoint(
        cfg.DIRS.SAVE_DIR,
        filename="ckpt_score_{val_score:0.4f}",
        monitor="val_score",
        mode="max",
        save_top_k=cfg.TRAIN.SAVE_TOP_K,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
        save_last=True,
    )

    check_point_dice = pl.callbacks.model_checkpoint.ModelCheckpoint(
        cfg.DIRS.SAVE_DIR,
        filename="ckpt_dice_{val_dice:0.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=cfg.TRAIN.SAVE_TOP_K,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
        save_last=True,
    )
    # Initialize a LearningRateMonitor callback to log the learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval=None)
    # Initialize a EarlyStopping callback to stop training if the validation loss does not improve for a certain number of epochs
    early_stopping = EarlyStopping(
        monitor="val_score",
        mode="max",
        patience=cfg.OPT.PATIENCE_ES,
        verbose=True,
        strict=False,
    )

    print("class_weight:", cfg.DATA.CLASS_WEIGHT)
    print("Train on fold:", cfg.TRAIN.FOLD)
    print("Use loss:", cfg.TRAIN.LOSS)

    # Define a dictionary with the parameters for the Trainer object
    PARAMS_TRAINER = {
        "accelerator": cfg.SYS.ACCELERATOR,
        "devices": cfg.SYS.DEVICES,
        "benchmark": True,
        "enable_progress_bar": True,
        # "overfit_batches" :5,
        "logger": wandb_logger,
        "callbacks": [check_point_score, check_point_dice, early_stopping, lr_monitor],
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 1,
        "max_epochs": cfg.TRAIN.EPOCHS,
        "precision": cfg.SYS.MIX_PRECISION,
    }

    # Initialize a Trainer object with the specified parameters
    trainer = pl.Trainer(**PARAMS_TRAINER)
    # Get a list of file paths for all non-hidden files in the SAVE_DIR directory
    checkpoint_paths = [cfg.DIRS.SAVE_DIR + f for f in os.listdir(cfg.DIRS.SAVE_DIR) if not f.startswith(".")]
    checkpoint_paths.sort()
    # If there are checkpoint paths and the load_checkpoint flag is set to True
    if checkpoint_paths and cfg.TRAIN.LOAD_CHECKPOINT:
        # Select the second checkpoint in the list (index 0)
        checkpoint = checkpoint_paths[cfg.TRAIN.IDX_CHECKPOINT]
        print(f"load checkpoint: {checkpoint}")
        # Load the model weights from the selected checkpoint
        segmenter = Segmenter.load_from_checkpoint(
            checkpoint_path=checkpoint,
            model=model,
            class_weight=cfg.DATA.CLASS_WEIGHT,
            num_classes=cfg.DATA.NUM_CLASS,
            learning_rate=cfg.OPT.LEARNING_RATE,
            factor_lr=cfg.OPT.FACTOR_LR,
            patience_lr=cfg.OPT.PATIENCE_LR,
        )

    # Train the model using the train_dataset and test_dataset data loaders
    trainer.fit(segmenter, train_dataset, test_dataset)

    wandb.finish()
