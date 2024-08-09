import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os

class CustomCIFAR10(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        # Assuming the directory structure is similar to CIFAR-10
        self.data_dir = os.path.join(root_dir, 'train' if train else 'test')
        self.classes = os.listdir(self.data_dir)

        self.image_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]
        return image, label

# Example usage
data_dir = 'path/to/your/custom/dataset'
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CustomCIFAR10(data_dir, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
