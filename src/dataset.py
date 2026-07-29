import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ToTensor, Normalize


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
    train_transforms = build_train_transformations(augmentation)
    train_transforms.extend([ToTensor(), Normalize(mean, std)])
    train_transformations = Compose(train_transforms)

    val_transformations = Compose([
        ToTensor(),
        Normalize(mean, std)
    ])

    return train_transformations, val_transformations

def build_train_transformations(augmentation):
    transform_map = {
        "random_horizontal_flip": lambda : RandomHorizontalFlip(p=float(augmentation['random_horizontal_flip']['p'])),
        "random_rotation": lambda : RandomRotation(degrees=int(augmentation['random_rotation']['degrees'])),
    }

    transforms = []
    for k in augmentation.keys():
        transforms.append(transform_map[k]())

    return transforms