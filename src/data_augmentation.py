import os
from typing import List, Tuple
from PIL import Image
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop, RandomVerticalFlip

def augment_images(
        image_list: List[str],
        label: int,
        augmentations: int,
        output_dir: str,
        transform: transforms = None
) -> List[Tuple[str, int]]:
    if transform is None:
        transform = transforms.Compose([
            RandomHorizontalFlip(),
            RandomRotation(10),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            RandomResizedCrop(224, scale=(0.8, 1.0)),
            RandomVerticalFlip()
        ])

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


def main():
    data_dir = '../data/dataset'
    augment_dir = '../data/dataset_augmented'

    ship_dir = os.path.join(data_dir, 'ship')
    sea_dir = os.path.join(data_dir, 'sea')
    ship_images_raw = [os.path.join(ship_dir, file_name) for file_name in os.listdir(ship_dir) if file_name.endswith(('.jpg', '.png'))]
    sea_images_raw = [os.path.join(sea_dir, file_name) for file_name in os.listdir(sea_dir) if file_name.endswith(('.jpg', '.png'))]

    augment_transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomRotation(30, fill=(128, 128, 128)),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        RandomResizedCrop(224, scale=(0.8, 1.0)),
        RandomVerticalFlip()
    ])

    os.makedirs(augment_dir, exist_ok=True)
    os.makedirs(os.path.join(augment_dir, 'ship'), exist_ok=True)
    os.makedirs(os.path.join(augment_dir, 'sea'), exist_ok=True)

    augmented_ship_images = augment_images(ship_images_raw, 1, 5, os.path.join(augment_dir, 'ship'), augment_transform)
    augmented_sea_images = augment_images(sea_images_raw, 0, 1, os.path.join(augment_dir, 'sea'), augment_transform)

if __name__ == '__main__':
    main()
