import os
import cv2


def create_output_folder(output_folder):
    """
    Tworzy folder wyjściowy, jeśli nie istnieje.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def process_image(img_path, crop_ratio=0.2):
    """
    Wczytuje obraz i obcina go, pozostawiając 25% szerokości od lewej strony.
    """
    img = cv2.imread(img_path)
    if img is not None:
        height, width = img.shape[:2]
        new_width = int(width * crop_ratio)
        cropped_img = img[:, :new_width]
        return cropped_img
    else:
        raise ValueError(f"Error reading {img_path}")


def save_image(output_path, image):
    """
    Zapisuje przetworzony obraz w podanej ścieżce.
    """
    cv2.imwrite(output_path, image)


def crop_images(input_folder, output_folder, crop_ratio=0.2):
    """
    Przetwarza wszystkie obrazy w folderze wejściowym i zapisuje je w folderze wyjściowym.
    """
    create_output_folder(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            try:
                cropped_img = process_image(img_path, crop_ratio)
                output_path = os.path.join(output_folder, filename)
                save_image(output_path, cropped_img)
                print(f"Processed {filename}")
            except ValueError as e:
                print(e)
        else:
            print(f"Skipping non-image file {filename}")


if __name__ == "__main__":
    input_folder = "04_input_index"
    output_folder = "04_output_index"
    crop_images(input_folder, output_folder)
