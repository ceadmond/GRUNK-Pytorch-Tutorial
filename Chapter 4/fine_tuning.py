import os

DATA_ROOT = "data"
os.environ["TORCH_HOME"] = DATA_ROOT
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

all_labels_df = pd.read_csv(os.path.join(DATA_ROOT, "labels.csv"))
all_labels_df.head()

breeds = all_labels_df.breed.unique()
breed2idx = dict((breed, idx) for idx, breed in enumerate(breeds))
idx2breed = dict((idx, breed) for idx, breed in enumerate(breeds))
# print(len(breeds))

all_labels_df["label_idx"] = [breed2idx[b] for b in all_labels_df.breed]
all_labels_df.head()


class DogDataset(Dataset):
    def __init__(self, labels_df, img_path, transform=None):
        self.labels_df = labels_df
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return self.labels_df.shape[0]

    def __getitem__(self, idx):
        image_name = os.path.join(self.img_path, self.labels_df.id[idx]) + ".jpg"
        img = Image.open(image_name)
        label = self.labels_df.label_idx[idx]

        if self.transform:
            img = self.transform(img)
        return img, label


IMG_SIZE = 224
BATCH_SIZE = 64
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_transforms = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ]
)

dataset_names = ["train", "valid"]
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
train_split_idx, val_split_idx = next(
    iter(stratified_split.split(all_labels_df.id, all_labels_df.breed))
)
train_df = all_labels_df.iloc[train_split_idx].reset_index()
val_df = all_labels_df.iloc[val_split_idx].reset_index()
# print(len(train_df))
# print(len(val_df))

image_transforms = {"train": train_transforms, "valid": val_transforms}

train_dataset = DogDataset(
    train_df, os.path.join(DATA_ROOT, "train"), transform=image_transforms["train"]
)
val_dataset = DogDataset(
    val_df, os.path.join(DATA_ROOT, "train"), transform=image_transforms["valid"]
)
image_dataset = {"train": train_dataset, "valid": val_dataset}

image_dataloader = {
    x: DataLoader(image_dataset[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    for x in dataset_names
}
dataset_sizes = {x: len(image_dataset[x]) for x in dataset_names}

model_ft = models.resnet50(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

# print(model_ft.fc)
num_fc_ftr = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_fc_ftr, len(breeds))
model_ft = model_ft.to(DEVICE)
# print(model_ft)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([{"params": model_ft.fc.parameters()}], lr=0.001)


def train(model, device, train_loader, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
    print("Train Epoch: {}\t Loss: {:.6f}".format(epoch, loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            test_loss += criterion(y_hat, y).item()
            pred = y_hat.max(1, keepdim=True)[1]
            correct += pred.eq(y.view(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%\n)".format(
            test_loss, correct, len(val_dataset), 100.0 * correct / len(val_dataset)
        )
    )


for epoch in range(1, 10):
    train(
        model=model_ft,
        device=DEVICE,
        train_loader=image_dataloader["train"],
        epoch=epoch,
    )
    test(model=model_ft, device=DEVICE, test_loader=image_dataloader["valid"])
