# UNet/dataset.py

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ProstateGlandDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames, image_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # This is the change: we now accept a list of filenames
        self.images = image_filenames
        
        self.image_resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image = self.image_resize(image)
        mask = self.mask_resize(mask)
        
        image = np.array(image)
        mask = np.array(mask, dtype=np.float32)
        mask[mask > 0] = 1.0

        image = self.to_tensor(image)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask