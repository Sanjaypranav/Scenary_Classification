import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import numpy as np

class ScenaryDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.scenary_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.scenary_frame) #17000 images

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.scenary_frame.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.scenary_frame.iloc[idx, 1]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label

