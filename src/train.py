import torch
import os
import glob
import numpy as np
from sklearn import (preprocessing, model_selection)
import config
import dataset,engine
from model import CaptchaModel


def run_training():
    image_files = glob.glob(os.path.abspath(
        os.path.join(config.DATA_DIR, "*.png")))
    labels = [list(x.split("/")[-1].split(".")[0]) for x in image_files]
    labels_flat = [c for x in labels for c in x]

    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(labels_flat)
    tar_enc = np.array([label_enc.transform(x) for x in labels]) + 1
    train_X, test_X, train_y, test_y, train_target, test_target = model_selection.train_test_split(
        image_files, tar_enc, labels)

    train_dataset = dataset.DataSet(
        train_X, 
        train_y, 
        resize = (config.IMG_HEIGHT, config.IMG_WIDTH))
    
    test_dataset = dataset.DataSet(
        test_X, 
        test_y, 
        resize = (config.IMG_HEIGHT, config.IMG_WIDTH))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = config.BATCH_SIZE, 
        num_workers= config.NUM_WORKERS, 
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS
    )

    cm = CaptchaModel(num_chars = len(label_enc.classes_))
    cm.to(config.DEVICE)

    optimizer=  torch.optim.Adam(cm.parameters(), lr = 3e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=10, verbose=True
    )

    for epoch in range(config.EPOCHS):

        train_loss = engine.train_fn(cm, train_dataloader, optimizer)


run_training()
