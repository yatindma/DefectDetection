import torchvision.datasets
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode  # validation or train
        # self.image_list = [os.path.join(path, name) for path, subdirs, files in os.walk() for name in files]
    pass

    def  __len__(self):
        if self.mode == "val":
            return len(self.train_data)
        elif self.mode == "train":
            return len(self.val_data)

    def transform(self, image, path):
        # Perform the transformation over the image data
        transformed_image = image
        transformer = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std),
            tv.transforms.RandomVerticalFlip(p=0.2)
            ])
        return Dataset(cfg_path=path, valid_split_ratio=0.2, transform=transformer, mode="train" if self.mode == "train" else "val")

    def __getitem__(self, index):
        # This function return the transformed image
        df = self.data.iloc[index]
        img_path = df['filename']
        crack = df["crack"]
        inactive = df['inactive']
        label = np.zeros(2, dtype=int)
        label[0] = int(crack)
        label[1] = int(inactive)
        image = imread(os.path.join(self.input_data_path, img_path))
        image = gray2rgb(image)
        image = self.transform(image, img_path)
        label = torch.from_numpy(label)
        return image, label


