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
                transforms.Resize((self.target_width, self.target_height)),
                transforms.ToTensor(),
                transforms.Normalize([0.5 for _ in range(self.channels)],
                                     [0.5 for _ in range(self.channels)]),
            ]
        )
        dataset = datasets.ImageFolder(root=src_dir, transform=transform)
        return dataset

    def resize_dataset(self, src_dir):
        for filename in os.listdir(src_dir):
            self.resize_image(src_dir+"/"+filename)

    def resize_image(self, image_path):
        image = cv2.imread(image_path)
        if self.is_too_small(image):
            resized_img = cv2.resize(image, (self.target_width, self.target_height),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = cv2.resize(image, (self.target_width, self.target_height),
                                     interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, resized_img)

    def is_too_small(self, image):
        height, width, channels = image.shape
        return height < self.target_height and width < self.target_width
