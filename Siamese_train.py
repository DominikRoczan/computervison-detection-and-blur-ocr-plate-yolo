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

# Definicja funkcji odległościowej
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

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
    # Ścieżki do obrazów (zamień na rzeczywiste ścieżki)
    image_paths = {
        "Poland1": "Pairs/01_Support Set/01_PL.jpg",
        "Germany1": "Pairs/01_Support Set/02_DE.jpg",
        "Holand1": "Pairs/01_Support Set/03_H.jpg",

        "Poland2": "Pairs/02_Query Set/04_PL.jpg",
        "Germany2": "Pairs/02_Query Set/05_DE.jpg",
        "Holand2": "Pairs/02_Query Set/06_H.jpg",

        "NoBlue": "Pairs/02_Query Set/07.jpg"
    }

    # Zbiór wsparcia (Support Set)
    support_set = [image_paths["Poland1"], image_paths["Germany1"], image_paths["Holand1"]]
    
    # Zbiór zapytań (Query Set)
    query_set = [image_paths["Poland2"], image_paths["Germany2"], image_paths["Holand2"], image_paths["NoBlue"]]

    # Tworzenie par
    positive_pairs = [
        (image_paths["Poland1"], image_paths["Poland2"]),
        (image_paths["Germany1"], image_paths["Germany2"]),
        (image_paths["Holand1"], image_paths["Holand2"])
    ]

    negative_pairs = [
        (image_paths["Poland1"], image_paths["Germany2"]),
        (image_paths["Poland1"], image_paths["Holand2"]),
        (image_paths["Germany1"], image_paths["Poland2"]),
        (image_paths["Germany1"], image_paths["Holand2"]),
        (image_paths["Holand1"], image_paths["Poland2"]),
        (image_paths["Holand1"], image_paths["Germany2"]),
        (image_paths["Poland1"], image_paths["NoBlue"]),
        (image_paths["Germany1"], image_paths["NoBlue"]),
        (image_paths["Holand1"], image_paths["NoBlue"])
    ]

    # Łączenie par i etykiet
    image_pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

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

def evaluate_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            output1, output2 = model(img0, img1)
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            print(f"Euclidean Distance: {euclidean_distance.item()}, Label: {label.item()}")

def main():
    # Przygotowanie danych
    image_pairs, labels = prepare_data()

    transform = transforms.Compose([transforms.Resize((105, 105)),
                                    transforms.ToTensor()])

    dataset = CustomDataset(image_pairs, labels, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

    # Ustawienie urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inicjalizacja modelu, straty i optymalizatora
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Trening modelu
    num_epochs = 150
    train_model(model, dataloader, criterion, optimizer, num_epochs, device)

    # Zapisanie modelu
    save_model(model, "Siamese_model")

    # Ewaluacja modelu
    test_image_pairs, test_labels = prepare_data()  # Zamień na rzeczywiste dane testowe
    test_dataset = CustomDataset(test_image_pairs, test_labels, transform=transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    evaluate_model(model, test_dataloader, device)

if __name__ == "__main__":
    main()
