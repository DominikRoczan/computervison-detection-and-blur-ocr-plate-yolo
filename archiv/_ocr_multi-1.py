import torch
import os
from PIL import Image
from strhub.data.module import SceneTextDataModule

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

# Ścieżka do folderu ze zdjęciami
folder_path = '/output_cropped'

# Iteracja przez wszystkie pliki w folderze
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Możesz dostosować rozszerzenia
        # Wczytaj obrazek
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')
        # Preprocessing
        img = img_transform(img).unsqueeze(0)
        # Predykcja
        logits = parseq(img)
        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)
        print(f'Image: {filename}, Decoded label = {label[0]}')
    else:
        continue
