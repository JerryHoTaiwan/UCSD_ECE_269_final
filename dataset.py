from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import cv2
import pandas as pd
import os

class ReweightDataset(Dataset):
    def __init__(self, imgpath, csv_path):
        self.imgpath = imgpath# Nx1x32x32x16
        csv = pd.read_csv(csv_path)
        self.labels = csv["label"]
        return
    
    def __len__(self):
        #return self.pad_vol.shape[0]
        return len(self.labels)
    
    def __getitem__(self, idx):
        path = os.path.join(self.imgpath, "reweight_{}.npy".format(idx))
        img = np.load(path)
        img = torch.FloatTensor(img)
        label = torch.from_numpy(np.array([self.labels[idx]]).astype(np.int64))
        return (img, label)