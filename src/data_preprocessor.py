import cv2
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class DataPreprocessor:
    def __init__(self, target_width, target_height, img_channels):
        self.target_width = target_width
        self.target_height = target_height
        self.channels = img_channels

    def load_dataset(self, src_dir):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),
                transforms.Resize((self.target_width, self.target_height)),
                transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(self.channels)],
                                     [0.5 for _ in range(self.channels)]),
            ]
        )
        dataset = datasets.ImageFolder(root=src_dir, transform=transform)
        return dataset
