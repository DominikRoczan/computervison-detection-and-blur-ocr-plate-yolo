import cv2
import pytesseract
from PIL import Image
import numpy as np

# Ścieżka do obrazu
image_path = 'input/300.jpg'

# Ustawienie ścieżki do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Otwarcie obrazu za pomocą PIL
image = Image.open(image_path)

# Konwersja obrazu do formatu numpy
image_np = np.array(image)

# Konwersja obrazu do skali szarości
gray = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

# Detekcja tekstu za pomocą funkcji detekcji tekstu z OpenCV
detections = pytesseract.image_to_data(image_np, output_type=pytesseract.Output.DICT, lang='pol')

# Tworzenie maski zawierającej obszary z tekstem
text_mask = np.zeros_like(gray)
for i in range(len(detections['text'])):
    if int(detections['conf'][i]) > 60:  # Ustal próg pewności tekstu
        x, y, w, h = detections['left'][i], detections['top'][i], detections['width'][i], detections['height'][i]
        cv2.rectangle(text_mask, (x, y), (x + w, y + h), (255), -1)

# Wykrywanie krawędzi za pomocą algorytmu Canny
edges = cv2.Canny(gray, 700, 100)

# Usunięcie krawędzi poza obszarem tekstu
edges[text_mask == 0] = 0

# Konwersja obrazu z formatu numpy na obiekt Image PIL
edges_image = Image.fromarray(edges)

# Ustawienie nowej wartości DPI
new_dpi = (300, 300)  # Nowa wartość DPI

# Eksportowanie i ponowne importowanie obrazu z nowymi wartościami DPI
edges_image_with_new_dpi_path = 'edges_102A.jpg'
edges_image.save(edges_image_with_new_dpi_path, dpi=new_dpi)
image_with_new_dpi = Image.open(edges_image_with_new_dpi_path)

# Wyświetlenie obrazu (opcjonalne)
image_with_new_dpi.show()

# Przetworzenie obrazu do tekstu za pomocą pytesseract
text = pytesseract.image_to_string(image_with_new_dpi, lang='pol')

# Wyświetlenie rozpoznanego tekstu
print('Tekst: ', text)

# Wydrukowanie informacji o obrazie
print("Rozmiar obrazu (szerokość x wysokość):", image_with_new_dpi.size)
print("Mode (tryb kolorów):", image_with_new_dpi.mode)
print("Format:", image_with_new_dpi.format)

# DPI obrazu po zmianie
dpi = image_with_new_dpi.info.get('dpi', "Brak informacji o DPI")
print("DPI:", dpi)
