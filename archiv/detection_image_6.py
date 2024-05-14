import os
from ultralytics import YOLO
import cv2
import easyocr

# Ścieżka do katalogu zawierającego zdjęcia
directory_path = '../input'

# Lista na ścieżki do zdjęć
image_paths = []

# Sprawdź wszystkie pliki w katalogu
for filename in os.listdir(directory_path):
    # Sprawdź, czy plik jest plikiem graficznym (np. jpg, png)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Jeśli tak, dodaj jego pełną ścieżkę do listy image_paths
        image_paths.append(os.path.join(directory_path, filename))

# Załaduj model YOLO
model_path = '../trained_model/weights/_best.pt'
model = YOLO(model_path)  # Używając ścieżki do wcześniej wytrenowanego modelu

# OCR
ocr = easyocr.Reader(['en'])

# Ścieżka do zapisania pliku tekstowego
dir_name = '../output'
if not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)

# Ścieżka do zapisania wyciętych obrazów
cropped_dir = '../input_cropped'
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir, exist_ok=True)

count = {}  # Słownik do śledzenia licznika dla każdego pliku

# Przetwarzaj wszystkie obrazy w katalogu
for image_path in image_paths:
    image = cv2.imread(image_path)

    # Przeprowadź detekcję na obrazie
    results = model(image)

    idx = 0
    detected = False  # Flaga, czy wykryto jakieś tablice
    cropped_images = []  # Lista na wycięte obrazy tablic

    # Przetwarzanie wyników detekcji
    for r in results:
        boxes = r.boxes
        class_names = list(r.names.values())

        # Iteruj przez każde pole (bounding box)
        for b in boxes:
            xyxy = b.xyxy  # Zakładamy, że xyxy to tensor z 4 wartościami [x1, y1, x2, y2]

            confidence = float(b.conf)  # Prawidłowa pewność detekcji
            class_id = b.cls.item()  # Get class ID for each box
            class_name = r.names[class_id]  # Convert class ID to class name
            label = f"{class_name} {confidence:.2f}"

            if confidence > 0.15:
                detected = True
                box_width = int(xyxy[0][2]) - int(xyxy[0][0])
                # Inkrementuj licznik dla aktualnego pliku
                count[image_path] = count.get(image_path, 0) + 1
                current_count = count[image_path]
                print(f'{current_count} Conf: {confidence:.2f}')

                # cropped plates
                cropped_image = image[int(xyxy[0][1]):int(xyxy[0][3]), int(xyxy[0][0]):int(xyxy[0][2])]
                cropped_images.append(cropped_image)

                # Zapisz wycięty obraz do folderu input_cropped
                cropped_name = f'{os.path.splitext(os.path.basename(image_path))[0]}_cropped_{current_count}.jpg'
                cv2.imwrite(os.path.join(cropped_dir, cropped_name), cropped_image)

                # Zablurowanie obszaru wewnątrz prostokąta
                x1, y1, x2, y2 = int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[0][2]), int(xyxy[0][3])
                cropped_region = image[y1:y2, x1:x2]
                blurred_region = cv2.blur(cropped_region, (23, 23))  # Rozmycie Gaussowskie

                # Wstawienie zblurowanego obszaru z powrotem do obrazu
                image[y1:y2, x1:x2] = blurred_region

    if detected:
        # Zapisywanie obrazu końcowego
        final_image_path = os.path.join(dir_name, f'{os.path.splitext(os.path.basename(image_path))[0]}_Blur.jpg')
        cv2.imwrite(final_image_path, image)

    else:
        # Obsługa przypadku, gdy nie wykryto tablic
        print(f"No license plates were detected in {image_path}.")
