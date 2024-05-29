import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2

# Definicja modelu Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, num_features=9216):
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
            nn.Linear(num_features, 4096),
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


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

def detect_license_plate(image_path, model, threshold=1.95):
    # Przetwarzanie obrazu
    transform = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0)

    # Wykrywanie tablicy rejestracyjnej przy użyciu modelu
    with torch.no_grad():
        output1 = model.forward_one(image_tensor)
        output1_sigmoid = torch.sigmoid(output1)
        max_value, _ = torch.max(output1_sigmoid, dim=1)
        if max_value.item() > threshold:
            return True
        else:
            return False





def draw_rectangle(image_path, output_path, model, threshold=0.5):
    # Wczytanie obrazu
    image = cv2.imread(image_path)

    # Detekcja niebieskiego kawałka tablicy rejestracyjnej
    if detect_license_plate(image_path, model, threshold):
        # Obrysowanie obszaru z niebieskim kawałkiem tablicy rejestracyjnej prostokątem
        cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), 2)  # Przykładowe współrzędne prostokąta

    # Zapisanie obrazu z zaznaczonym obszarem
    cv2.imwrite(output_path, image)

def main():
    # Ścieżka do zapisanego modelu
    model_path = "Siamese_model/siamese_model.pth"

    # Inicjalizacja modelu
    model = SiameseNetwork()
    load_model(model, model_path)

    # Ścieżka do obrazu wejściowego
    image_path = "Pairs/02_Query Set/XX.jpg"

    # Ścieżka do obrazu wyjściowego
    output_path = "Pairs/02_Query Set/XX.jpg"

    # Wykrywanie i zaznaczanie niebieskiego kawałka tablicy rejestracyjnej na obrazie
    draw_rectangle(image_path, output_path, model)

    # Wyświetlenie obrazu wyjściowego
    image = cv2.imread(output_path)
    cv2.imshow("Detected License Plate", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
