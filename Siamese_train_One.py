import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Definicja modelu Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        x = self.cnn1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Przygotowanie zbioru danych
class CustomDataset(Dataset):
    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img1 = Image.open(self.image_pairs[index][0]).convert('L')
        img2 = Image.open(self.image_pairs[index][1]).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([int(self.labels[index])], dtype=np.float32))

    def __len__(self):
        return len(self.image_pairs)

def prepare_data():
    # Ścieżki do obrazów
    image_paths = {
        "PL1": "Pairs/01_Support Set/01_PL.jpg",
        "PL2": "Pairs/01_Support Set/02_PL.jpg",
        "PL3": "Pairs/01_Support Set/03_PL.jpg",
    }

    # Zbiór wsparcia (Support Set)
    support_set = [image_paths["PL1"], image_paths["PL2"], image_paths["PL3"]]

    # Tworzenie par
    positive_pairs = [
        (image_paths["PL1"], image_paths["PL2"]),
        (image_paths["PL1"], image_paths["PL3"]),
        (image_paths["PL2"], image_paths["PL3"])
    ]

    # Łączenie par i etykiet
    image_pairs = positive_pairs
    labels = [1] * len(positive_pairs)

    return image_pairs, labels

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{len(dataloader)}, Loss: {loss_contrastive.item()}")
def save_model(model, folderpath):
    """
    Zapisuje stan modelu do pliku.

    Args:
        model (torch.nn.Module): Model do zapisania.
        folderpath (str): Ścieżka do folderu, gdzie ma być zapisany model.
    """
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    filepath = os.path.join(folderpath, "siamese_model.pth")
    torch.save(model.state_dict(), filepath)

def main():
    # Przygotowanie danych
    image_pairs, labels = prepare_data()

    transform = transforms.Compose([transforms.Resize((105, 105)),
                                    transforms.ToTensor()])

    dataset = CustomDataset(image_pairs, labels, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=6)

    # Ustawienie urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inicjalizacja modelu, straty i optymalizatora
    model = SiameseNetwork().to(device)
    criterion = nn.MarginRankingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Trening modelu
    num_epochs = 70
    train_model(model, dataloader, criterion, optimizer, num_epochs, device)

    # Zapisanie modelu
    save_model(model, "Siamese_model")

if __name__ == "__main__":
    main()
