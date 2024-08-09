from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import os


data_dir = 'pattern-recognition'

print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 (adjust as needed)
    transforms.ToTensor()  # Convert to tensor
])

dataset = ImageFolder(data_dir+'/train', transform=transform)
img, label = dataset[0]
print(img.shape, label)
print(dataset.classes)
print(len(dataset.classes))


print(len(dataset))
val_size = 500
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)

batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)