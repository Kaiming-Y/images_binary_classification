import os
import random
from PIL import Image
from typing import Tuple, List
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomRotation


def load_data(
        data_dir: str,
        augment_dir: str,
        batch_size: int = 32,
        test_ratio: float = 0.2,
        augment: bool = False
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    augment_transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomRotation(180)
    ])

    # Original dataset
    ship_dir = os.path.join(data_dir, 'ship')
    sea_dir = os.path.join(data_dir, 'sea')
    ship_images_raw = [os.path.join(ship_dir, file_name) for file_name in os.listdir(ship_dir)]
    sea_images_raw = [os.path.join(sea_dir, file_name) for file_name in os.listdir(sea_dir)]
    ship_images = [(img, 1) for img in ship_images_raw]
    sea_images = [(img, 0) for img in sea_images_raw]

    if augment:
        # Augmented dataset
        os.makedirs(augment_dir, exist_ok=True)
        os.makedirs(os.path.join(augment_dir, 'ship'), exist_ok=True)
        os.makedirs(os.path.join(augment_dir, 'sea'), exist_ok=True)
        augmented_ship_images = augment_images(ship_images_raw, 1, 5, os.path.join(augment_dir, 'ship'), augment_transform)
        augmented_sea_images = augment_images(sea_images_raw, 0, 1, os.path.join(augment_dir, 'sea'), augment_transform)

        # Integrate original and augmented data sets: 6000 ship & 6000 sea images
        ship_images = ship_images + augmented_ship_images
        sea_images = sea_images + augmented_sea_images

    # Partition the data set
    random.shuffle(ship_images)
    random.shuffle(sea_images)
    ship_size = len(ship_images)
    sea_size = len(sea_images)
    ship_test_size = int(ship_size * test_ratio)
    sea_test_size = int(sea_size * test_ratio)
    ship_train_images = ship_images[ship_test_size:]
    ship_test_images = ship_images[:ship_test_size]
    sea_train_images = sea_images[sea_test_size:]
    sea_test_images = sea_images[:sea_test_size]

    train_images = ship_train_images + sea_train_images
    test_images = ship_test_images + sea_test_images


    class CustomDataset(Dataset):
        def __init__(
                self,
                image_list: List[Tuple[str, int]],
                transform=None
        ) -> None:
            super(CustomDataset, self).__init__()
            self.image_list = image_list
            self.transform = transform

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, idx):
            img_path, label = self.image_list[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    train_dataset = CustomDataset(train_images, transform)
    test_dataset = CustomDataset(test_images, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def augment_images(
        image_list: List[str],
        label: int,
        augmentations: int,
        output_dir: str,
        transform: transforms = None
) -> List[Tuple[str, int]]:
    if transform is None:
        transform = transforms.Compose([RandomHorizontalFlip(), RandomRotation(180)])

    augmented_images = []
    for img_path in image_list:
        img = Image.open(img_path)
        base_name = os.path.basename(img_path)
        for i in range(augmentations):
            augmented_img = transform(img)
            augmented_img_path = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
            augmented_img.save(augmented_img_path)
            augmented_images.append((augmented_img_path, label))
    return augmented_images
