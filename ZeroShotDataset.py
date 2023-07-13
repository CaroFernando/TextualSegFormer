import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torchvision
import cv2
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

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
            # keep only classes that are in the unseen classes list
            self.df = self.df[self.df["label"].isin(self.unseen_clases)]

        elif self.filter_seen:
            # keep only classes that are not in the unseen classes list
            self.df = self.df[~self.df["label"].isin(self.unseen_clases)]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_folder + row["image"] 
        mask_path = self.mask_folder + row["mask"] 

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_processor(images = image, return_tensors="pt")['pixel_values'][0]
        mask = self.mask_transform(mask)

        label = row["label"]
        template = self.templates[np.random.randint(len(self.templates))]
        text = template.format(label)
        text = self.tokenizer.encode(text)
        text = torch.Tensor(text).long()

        return image, text, mask
    
    def collate_fn(self, batch):
        images, texts, masks = zip(*batch)
        images = torch.stack(images)
        masks = torch.stack(masks)
        texts = pad_sequence(texts, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return images, texts.long(), masks
