import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np


class SemSegCompose(nn.Module):
    def __init__(self, padding, mean_ch, std_ch, angle):
        super().__init__()
        self.pad = T.Pad(padding)
        self.norm = T.Normalize(mean_ch, std_ch)
        self.angle = angle
        
    def forward(self, img, mask_img):
        img, mask_img = self.pad(img), self.pad(mask_img)
        
        # random vertical flip
        if np.random.random() > 0.5:
            img, mask_img = TF.vflip(img), TF.vflip(mask_img)
            
        # random horizontal flip
        if np.random.random() > 0.5:
            img, mask_img = TF.hflip(img), TF.hflip(mask_img)
        
        # random rotation
        if np.random.random() > 0.5:
            angle = np.random.randint(-self.angle, self.angle)
            img, mask_img = TF.rotate(img, angle), TF.rotate(mask_img, angle)
            
        img = self.norm(img)
        return img, mask_img