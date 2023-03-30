import torch
from torch.utils import data
from torchvision import transforms
import cv2
import os
import albumentations as A
import random

class Dataset(data.Dataset):
    def __init__(self, class_list, path_list, img_size):
        super().__init__()
        self.class_list = class_list
        self.img_size = img_size
        self.paths = path_list

        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda t: (t * 2) - 1)
        # ])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        elastictransform = A.ElasticTransform(alpha=1, 
                                            sigma=10, 
                                            alpha_affine=20, 
                                            interpolation=1, 
                                            border_mode=0, 
                                            value=None, 
                                            mask_value=None, 
                                            always_apply=False, 
                                            approximate=False, 
                                            same_dxdy=True, 
                                            p=0.5)

        resize = A.Resize(height=img_size, width=img_size, p=1)

        aug_list = [elastictransform, resize]

        self.compose = A.Compose(aug_list)

    def __getitem__(self, index):
        f = self.paths[index]
        name = f.split('/')[2]

        label = self.class_list.index(name)

        img = cv2.imread(f)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w= img.shape[0], img.shape[1]
        img = cv2.resize(img,(self.img_size, self.img_size))
        # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        otu_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,otu_img = cv2.threshold(otu_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        otu_img = cv2.cvtColor(otu_img, cv2.COLOR_GRAY2RGB)

        img = cv2.bitwise_or(otu_img, img)

        if int(random.uniform(0,1)+0.5):
            img = cv2.bitwise_not(img)
            transformed = self.compose(image=img)
            img = transformed["image"]
            img = cv2.bitwise_not(img)

        return self.transform(img), torch.tensor(label)

    def __len__(self):
        return len(self.paths)