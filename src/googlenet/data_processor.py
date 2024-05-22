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
        test_ratio: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    augment_transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomRotation(10)
    ])

    # Original dataset
    ship_dir = os.path.join(data_dir, 'ship')
    sea_dir = os.path.join(data_dir, 'sea')
    ship_images = [os.path.join(ship_dir, file_name) for file_name in os.listdir(ship_dir)]
    sea_images = [os.path.join(sea_dir, file_name) for file_name in os.listdir(sea_dir)]
    original_ship_images = [(img, 1) for img in ship_images]
    original_sea_images = [(img, 0) for img in sea_images]

    # Augmented dataset
    os.makedirs(augment_dir, exist_ok=True)
    os.makedirs(os.path.join(augment_dir, 'ship'), exist_ok=True)
    os.makedirs(os.path.join(augment_dir, 'sea'), exist_ok=True)
    # Augment ship dataset only
    augmented_ship_images = augment_images(ship_images, 1, 3, os.path.join(augment_dir, 'ship'), augment_transform)

    # Integrate and shuffle original and augmented data sets: 4000 ship + 3000 sea images
    all_images = original_ship_images + augmented_ship_images + original_sea_images
    random.shuffle(all_images)

    # Partition the data set
    total_size = len(all_images)
    test_size = int(total_size * test_ratio)
    train_images = all_images[test_size:]
    test_images = all_images[:test_size]

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
        transform = transforms.Compose([RandomHorizontalFlip(), RandomRotation(10)])

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
