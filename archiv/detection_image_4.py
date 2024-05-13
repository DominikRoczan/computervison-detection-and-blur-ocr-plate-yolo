import os
from ultralytics import YOLO
import cv2
import pytesseract
from PIL import Image

# Ustaw ścieżkę do pliku wykonywalnego tesseract, jeśli jest to konieczne
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Ścieżka do katalogu zawierającego zdjęcia
directory_path = '../input'

# Lista na ścieżki do zdjęć
image_paths = []

# Sprawdź wszystkie pliki w katalogu
for filename in os.listdir(directory_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_paths.append(os.path.join(directory_path, filename))

# Załaduj model YOLO
model_path = '../trained_model/weights/best.pt'
model = YOLO(model_path)

# Ścieżka do zapisania pliku tekstowego
dir_name = '../output'
if not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)

# Przetwarzaj wszystkie obrazy w katalogu
for image_path in image_paths:
    image = cv2.imread(image_path)
    results = model(image)

    detected = False
    cropped_images = []

    for r in results:
        boxes = r.boxes
        for b in boxes:
            xyxy = b.xyxy
            confidence = float(b.conf)
            class_id = b.cls.item()
            class_name = r.names[class_id]

            if confidence > 0.45:
                detected = True
                cropped_image = image[int(xyxy[0][1]):int(xyxy[0][3]), int(xyxy[0][0]):int(xyxy[0][2])]
                cropped_images.append(cropped_image)
                print(f'Conf: {confidence:.2f}')

                ocr_output_path = os.path.join(dir_name, f'ocr_output_{os.path.basename(image_path)}.txt')
                with open(ocr_output_path, 'w') as f:
                    for cropped_img in cropped_images:
                        # Konwersja fragmentu obrazu na obiekt Image PIL, który jest wymagany przez pytesseract
                        pil_img = Image.fromarray(cropped_img)
                        ocr_result = pytesseract.image_to_string(pil_img, lang='pol')
                        f.write('Detected text:\n' + ocr_result + '\n\n')

                '''x1, y1, x2, y2 = map(int, xyxy[0])
                cropped_region = image[y1:y2, x1:x2]
                blurred_region = cv2.blur(cropped_region, (23, 23))
                image[y1:y2, x1:x2] = blurred_region'''

    if detected:
        final_image_path = os.path.join(dir_name, f'{os.path.splitext(os.path.basename(image_path))[0]}_Blur.jpg')
        cv2.imwrite(final_image_path, image)
    else:
        print(f"No license plates were detected in {image_path}.")
