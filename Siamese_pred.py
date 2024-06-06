import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np


# Definicja modelu Siamese Network (taka sama jak podczas treningu)
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


# Funkcja do ładowania obrazu i przekształcania go
def load_and_transform_image(image_path, transform):
    image = Image.open(image_path).convert('L')
    return transform(image).unsqueeze(0)


# Funkcja do przewidywania niebieskiego tła
def predict_blue_background(model, support_image_path, query_image_path, device):
    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    support_image = load_and_transform_image(support_image_path, transform)
    query_image = load_and_transform_image(query_image_path, transform)

    support_image = support_image.to(device)
    query_image = query_image.to(device)

    model.eval()
    with torch.no_grad():
        output1, output2 = model(support_image, query_image)
        euclidean_distance = nn.functional.pairwise_distance(output1, output2).item()

    return euclidean_distance


# Funkcja do obrysowywania i wyświetlania obrazu
def draw_rectangle_and_display(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Załóżmy, że niebieskie tło znajduje się w górnej części obrazu
    top_left = (0, 0)
    bottom_right = (width, int(height * 0.2))  # Zmieniamy wymiary prostokąta według potrzeb

    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
    cv2.imshow('Detected Blue Background', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Główna funkcja do wykonywania predykcji
def main():
    # Ścieżki do modelu i obrazów (zamień na rzeczywiste ścieżki)
    model_path = "Siamese_model/siamese_model.pth"
    support_image_path = "Pairs/01_Support Set/01_PL.jpg"  # Obraz referencyjny z niebieskim tłem
    query_image_path = "Pairs/02_Query Set/XX.jpg"  # Obraz do sprawdzenia

    # Ustawienie urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ładowanie modelu
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Przewidywanie
    distance = predict_blue_background(model, support_image_path, query_image_path, device)
    print(f"Euclidean Distance: {distance}")

    # Ustalenie progu do klasyfikacji (próg powinien być dobrany na podstawie walidacji modelu)
    threshold = 1.0
    if distance < threshold:
        print("Niebieskie tło wykryte.")
        draw_rectangle_and_display(query_image_path)
    else:
        print("Niebieskie tło niewykryte.")


if __name__ == "__main__":
    main()
