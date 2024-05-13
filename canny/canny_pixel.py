import cv2
import numpy as np
import os

# Wczytaj obraz w skali szarości
image_gray = cv2.imread('input_cropped/12_p_ev_proxy_cropped_1.jpg', cv2.IMREAD_GRAYSCALE)

# Określ minimalną liczbę pikseli w najkrótszym wymiarze
min_dimension = 200

# Oblicz współczynnik skalowania
scale_factor = max(1, min_dimension / min(image_gray.shape[:2]))

# Oblicz wymagane DPI
required_dpi = 100
required_width = int(min_dimension * required_dpi / 25.4)
scale_factor_dpi = required_width / image_gray.shape[1]

# Powiększ obraz
resized_image_gray = cv2.resize(image_gray, None, fx=scale_factor * scale_factor_dpi, fy=scale_factor * scale_factor_dpi, interpolation=cv2.INTER_LINEAR)

# Pobierz rozmiar obrazu
height, width = resized_image_gray.shape

# Stwórz macierz pikseli
pixel_matrix = np.zeros((height, width), dtype=np.uint8)

# Wypełnij macierz wartościami pikseli
for y in range(height):
    for x in range(width):
        pixel_value = resized_image_gray[y, x]
        pixel_matrix[y, x] = pixel_value

# Utwórz nowy obraz, na którym będą wyświetlane wartości pikseli jako cyfry
image_with_text = np.zeros((height, width, 3), dtype=np.uint8)

# Umieść cyfry na obrazie na podstawie wartości pikseli w macierzy pikseli
for y in range(height):
    for x in range(width):
        # Konwertuj wartość piksela na string i wyświetl ją w odpowiednim miejscu na obrazie
        pixel_value = pixel_matrix[y, x]
        cv2.putText(image_with_text, str(pixel_value), (x*10, y*10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

# Wyświetl obrazek
cv2.imshow('Pixel Matrix', image_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Zapisz obraz w folderze wyjściowym
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'pixel_matrix.jpg')
cv2.imwrite(output_path, image_with_text, [cv2.IMWRITE_JPEG_QUALITY, 100])
print(f'Obraz został zapisany jako {output_path}')
