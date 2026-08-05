import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ToTensor, Normalize, RandomCrop, \
    RandomErasing, PILToTensor, ConvertImageDtype, ColorJitter


class CifarTestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.ids = sorted(int(path.stem) for path in self.test_dir.glob("*.png"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = self.retrieve_image(image_id)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_id

    def retrieve_image(self, image_id):
        image_path = self.test_dir / f"{image_id}.png"
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return image


class TmpCifarDataset(Dataset):
    def __init__(self, train_lib_dir, df, transform=None):
        self.train_lib_dir = train_lib_dir
        self.df = df
        self.transform = transform
        self.labels = df.label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image = self.retrieve_image(item)
        if self.transform is not None:
            image = self.transform(image)
        return image, item['target']

    def retrieve_image(self, item):
        image_path = os.path.join(self.train_lib_dir, item['file_name'])
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return image

    def get_label_description(self, idx):
        return self.df.iloc[idx]['label']

def define_transformations(augmentation, mean, std):
    train_transforms = build_train_transformations(augmentation, mean, std)
    train_transformations = Compose(train_transforms)

    val_transformations = Compose([
        ToTensor(),
        Normalize(mean, std)
    ])

    return train_transformations, val_transformations

def build_train_transformations(augmentation, mean, std):
    transform_map = {
        "random_horizontal_flip": lambda : RandomHorizontalFlip(p=float(augmentation['random_horizontal_flip']['p'])),
        "random_rotation": lambda : RandomRotation(degrees=int(augmentation['random_rotation']['degrees'])),
        "random_crop": lambda : RandomCrop(size=int(augmentation['random_crop']['size']), padding=int(augmentation['random_crop']['padding'])),
        "random_erasing": lambda : RandomErasing(),
        "pil_to_tensor": lambda : PILToTensor(),
        "convert_image_dtype": lambda : ConvertImageDtype(torch.float),
        "to_tensor": lambda : ToTensor(),
        "normalize": lambda : Normalize(mean=mean, std=std),
        "color_jitter": lambda : ColorJitter(brightness=.5, hue=.3)
    }

    transforms = []
    for k in augmentation.keys():
        transforms.append(transform_map[k]())

    return transforms