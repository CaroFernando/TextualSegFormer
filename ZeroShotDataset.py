import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision
import cv2

class ZeroShotDataset(Dataset):
    def __init__(
            self, 
            df, 
            image_folder,
            mask_folder,
            mask_size,
            templates, 
            unseen_clases, 
            image_processor, 
            tokenizer, 
            filter_unseen = True,
            filter_seen = False
        ):
        self.df = df
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.templates = templates
        self.unseen_clases = unseen_clases
        
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.filter_unseen = filter_unseen
        self.filter_seen = filter_seen

        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((mask_size, mask_size)),
            torchvision.transforms.ToTensor()
        ])

        if self.filter_unseen:
            self.df = self.df[self.df["label"].isin(self.unseen_clases)]

        elif self.filter_seen:
            self.df = self.df[~self.df["label"].isin(self.unseen_clases)]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_folder + row["image"] + ".jpg"
        mask_path = self.mask_folder + row["mask"] + ".png"

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = self.image_processor(images = image, return_tensors="pt")['pixel_values']
        mask = self.mask_transform(mask)

        label = row["label"]
        template = self.templates[np.random.randint(len(self.templates))]
        text = template.format(label)
        text = self.tokenizer.encode(text)

        return image, text, label
