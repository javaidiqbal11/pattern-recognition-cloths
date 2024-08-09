import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

mps_device = torch.device("cpu")


def train_model(model, train_loader, validation_loader, optimizer, n_epochs=20):
    # Global variable
    accuracy_list = []
    train_loss_list = []
    model = model.to(mps_device)
    train_cost_list = []
    val_cost_list = []

    for epoch in range(n_epochs):
        train_COST = 0
        for x, y in train_loader:
            x = x.to(mps_device)
            y = y.to(mps_device)
            model.train()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            train_COST += loss.item()

        train_COST = train_COST / len(train_loader)
        train_cost_list.append(train_COST)
        correct = 0

        # Perform the prediction on the validation data
        val_COST = 0
        for x_test, y_test in validation_loader:
            model.eval()
            x_test = x_test.to(mps_device)
            y_test = y_test.to(mps_device)
            z = model(x_test)
            val_loss = criterion(z, y_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
            val_COST += val_loss.item()

        val_COST = val_COST / len(validation_loader)
        val_cost_list.append(val_COST)

        accuracy = correct / val_size
        accuracy_list.append(accuracy)

        print("--> Epoch Number : {}".format(epoch + 1),
              " | Training Loss : {}".format(round(train_COST, 4)),
              " | Validation Loss : {}".format(round(val_COST, 4)),
              " | Validation Accuracy : {}%".format(round(accuracy * 100, 2)))

    return accuracy_list, train_cost_list, val_cost_list


def resnet_34():
    # Define the resnet model
    resnet = torchvision.models.resnet34(pretrained=True)

    # Update the fully connected layer of resnet with our current target of 10 desired outputs
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 13)

    # Initialize with xavier uniform
    torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet


model_mmtv6 = resnet_34()

data_dir = 'pattern-recognition'

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 224x224 (adjust as needed)
    transforms.ToTensor()  # Convert to tensor
])
dataset = ImageFolder(data_dir+'/train', transform=transform)
img, label = dataset[0]
print(img.size, label)
print(dataset.classes)


print(len(dataset))
total_size = len(dataset)
val_size = int(0.2 * total_size)
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)


# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model_mmtv6.parameters(), lr=learning_rate, momentum=0.2)

# Define the Scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min')

# Train the model
accuracy_list_normalv5, train_cost_listv5, val_cost_listv5 = train_model(model=model_mmtv6,
                                                                         n_epochs=25,
                                                                         train_loader=train_dl,
                                                                         validation_loader=val_dl,
                                                                         optimizer=optimizer)

print(accuracy_list_normalv5)

model_mmtv6.eval()
