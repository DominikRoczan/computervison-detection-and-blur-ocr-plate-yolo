import cv2
from PIL import Image
import numpy as np
import os

# Ścieżka do obrazu
image_path = 'output/21_p_ev_proxy_cropped_1.jpg'


# Otwarcie obrazu za pomocą PIL
image = Image.open(image_path)

# Minimalna szerokość dla obrazu
min_width = 300

# Obliczenie proporcji dla minimalnej szerokości
wpercent = (min_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(wpercent)))

# Skalowanie obrazu do minimalnej szerokości, zachowując proporcje
image = image.resize((min_width, hsize))

# Konwersja obrazu do formatu numpy
image_np = np.array(image)

# Konwersja obrazu do skali szarości
gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# Wykrywanie krawędzi za pomocą algorytmu Canny
edges = cv2.Canny(gray, 60, 655)  # Parametry: obraz wejściowy, wartość minimalna progowa, wartość maksymalna progowa

# Wykrywanie linii za pomocą transformacji Hougha
lines = cv2.HoughLines(edges, 7, np.pi/10, threshold=355)  # Parametry: obraz wejściowy, rozdzielczość r, c, próg

# Rysowanie wykrytych linii na obrazie
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)


# Wykrywanie krawędzi za pomocą algorytmu Canny
edges = cv2.Canny(image_np,100, 455)  # Parametry: obraz wejściowy, wartość minimalna progowa, wartość maksymalna progowa



# Konwersja obrazu z formatu numpy na obiekt Image PIL
edges_image = Image.fromarray(edges)

# Ustawienie nowej wartości DPI
new_dpi = (300, 300)  # Nowa wartość DPI

# Eksportowanie i ponowne importowanie obrazu z nowymi wartościami DPI
output_folder = 'output_canny'
os.makedirs(output_folder, exist_ok=True)  # Stworzenie folderu, jeśli nie istnieje
filename, file_extension = os.path.splitext(image_path)
edges_image_with_new_dpi_path = os.path.join(output_folder, filename.split('/')[-1] + '_edges' + file_extension)
edges_image.save(edges_image_with_new_dpi_path, dpi=new_dpi)

# Wyświetlenie obrazu (opcjonalne)
edges_image.show()

# Wydrukowanie informacji o obrazie
print("Rozmiar obrazu (szerokość x wysokość):", edges_image.size)
print("Mode (tryb kolorów):", edges_image.mode)
print("Format:", edges_image.format)

# DPI obrazu po zmianie
dpi = edges_image.info.get('dpi', "Brak informacji o DPI")
print("DPI:", dpi)
