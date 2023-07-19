from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

class PhraseCutDataset(Dataset):
    def __init__(self, annotations, image_folder, mask_folder, tokenizer, image_size=256, mask_size=64):
        self.annotations = annotations
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.tokenizer = tokenizer

        self.image_size = image_size
        self.mask_size = mask_size

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.image_inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((self.mask_size, self.mask_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = self.image_folder + str(self.annotations.iloc[index]['image']) + '.jpg'
        mask_path = self.mask_folder + str(self.annotations.iloc[index]['mask']) + '.jpg'
        phrase = self.annotations.iloc[index]['phrase']
        phrase = self.tokenizer.encode(phrase, add_special_tokens=True)
        phrase = torch.tensor(phrase).long()

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask, phrase
    
    def collate_fn(self, batch):
        images, masks, descriptions = zip(*batch)
        images = torch.stack(images)
        masks = torch.stack(masks)
        descriptions = pad_sequence(descriptions, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return images, masks, descriptions


