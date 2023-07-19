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
            image_size,
            mask_size,
            templates, 
            unseen_classes, 
            image_processor, 
            tokenizer, 
            filter_unseen = True,
            filter_seen = False
        ):
        self.df = df
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.templates = templates
        self.unseen_classes = unseen_classes
        
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.filter_unseen = filter_unseen
        self.filter_seen = filter_seen

        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            # Calculated mean and std
            # [0.46955295 0.44641594 0.4071987 ] [0.23915376 0.23444681 0.23751133]
            torchvision.transforms.Normalize(   
                mean=[0.46955295, 0.44641594, 0.4071987 ],
                std=[0.23915376, 0.23444681, 0.23751133]
            )
        ])

        self.mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((mask_size, mask_size)),
            torchvision.transforms.ToTensor()
        ])

        if self.filter_unseen:
            # keep only classes that are in the unseen classes list
            self.df = self.df[self.df["label"].isin(self.unseen_classes)]

        elif self.filter_seen:
            # keep only classes that are not in the unseen classes list
            self.df = self.df[~self.df["label"].isin(self.unseen_classes)]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_folder + row["image"] 
        mask_path = self.mask_folder + row["mask"] 

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        s_image = self.image_transform(image)
        clip_image = self.image_processor(images = image, return_tensors="pt")['pixel_values'][0]
        mask = self.mask_transform(mask)

        label = row["label"]
        template = self.templates[np.random.randint(len(self.templates))]
        text = template.format(label)
        text = self.tokenizer.encode(text)
        text = torch.Tensor(text).long()

        return s_image, clip_image, text, mask
    
    def collate_fn(self, batch):
        s_images, clip_images, texts, masks = zip(*batch)
        s_images = torch.stack(s_images)
        clip_images = torch.stack(clip_images)
        texts = pad_sequence(texts, batch_first=True)
        masks = torch.stack(masks)

        return s_images, clip_images, texts, masks
