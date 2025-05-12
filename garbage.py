import os
import cv2
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import h5py

hdf5_PATH = os.path.join(os.getcwd(), "data\\garbage_data.hdf5")

with h5py.File(hdf5_PATH, 'r') as hf:
    train_data = hf["train_data"][:]
    train_label = hf["train_label"][:]
    test_data = hf["test_data"][:]
    test_label = hf["test_label"][:]

classes = pd.read_csv("data\\label.csv")


class CustomImageDataset(Dataset):

    def __init__(self, image, label=None, classes=None, transform=None, target_transform=None, test=False):
        self.image = image
        self.label = label
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
        self.test = test

    def __len__(self):
        if not(self.test):
            return len(list(zip(self.image, self.label)))
        else:
            return len(self.image)
        
    def __getitem__(self, idx):
        if not (self.test):
            image = self.image[idx]
            label = self.label[idx]

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            return image, label
        else:
            image = self.image[idx]

            if self.transform:
                image = self.transform(image)
            return image
        
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, ), (.5, ))
])

train_dataset = CustomImageDataset(
    train_data,
    train_label,
    classes,
    transform=transform,
    test=False
)

test_dataset = CustomImageDataset(
    test_data,
    test_label,
    classes,
    transform=transform,
    test=True
)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 224 -> 112

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 112 -> 56

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 56 -> 28
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 28 -> 14

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 14 -> 7

            nn.Flatten(), # Flatten은 layer로 안쳐준다.
            nn.Dropout(.5),
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=10)
        )

    def forward(self, x):
        return self.conv_layer(x)
    
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = VGG16().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-5)

def train(model, train_dataloader, optimizer):
    model.train()
    correct = 0
    train_loss = 0
    total = len(train_dataloader)
    
    for batch, (image, label) in enumerate(train_dataloader):
        image = image.to(device)
        label = label.to(device).long()

        optimizer.zero_grad()
        pred_label = model(image)

        _, predicted = torch.max(pred_label.data, 1)
        correct += (predicted == label).sum().item()

        loss = criterion(pred_label, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if not(batch % 100):
            print(f"Train Epoch : {batch * len(image)} / {total} | Train Loss : {loss.item():.4f}")

    train_loss /= total
    correct = 100 * correct / total

    return correct, train_loss

class BestScoreSave():

    def __init__(self, model_name, mode="accuracy", delta=0.0, save_mode="weight", vervose=True, save=False):
        self.best_score = 0 if mode == "accuracy" else np.inf
        self.mode = mode
        self.save_mode = save_mode
        self.delta = delta
        self.vervose = vervose
        self.save = save
        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), self.model_name)
        if not (os.path.isdir(self.model_path)):
            os.mkdir(self.model_path)
            
    def __call__(self, score):
        if self.mode == "accuracy":
            if score > (self.best_score + self.delta):
                self.save = True
                self.best_score = score
            else:
                self.save = False
        elif self.mode == " loss":
            if score < (self.best_score - self.delta):
                self.save = True
                self.best_score = score
            else:
                self.save = False

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode='min', verbose=True):
        """
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
        """
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            self.early_stop = True
        else:
            self.early_stop = False

epochs = 10
bss = BestScoreSave(model_name="VGG16")
es = EarlyStopping()
losses = []
corrects = []
best_score = 0
for epoch in range(epochs):
    correct, train_loss = train(model, train_dataloader, optimizer)
    losses.append(train_loss)
    corrects.append(correct)
    print(f"Epoch : {epoch + 1} | Train Accuracy : {correct:.2f}")
    
    file_path = os.path.join(bss.model_path, f"{bss.model_name}_{bss.save_mode}_score{correct:.4f}_epoch{epoch + 1}.pt")
    bss(correct)
    if bss.save:
        if bss.save_mode == "weight":
            torch.save(model.state_dict(), file_path)
            if bss.vervose:
                print(f"[Saved Model Weight] Epoch: {epoch + 1} | Score: {correct}")
        elif bss.save_mode == "model":
            torch.save(model, file_path)
            if bss.vervose:
                print(f"[Saved Model] Epoch: {epoch + 1} | Score: {correct}")
    else:
        if bss.vervose:
            print("[Not Saved]")