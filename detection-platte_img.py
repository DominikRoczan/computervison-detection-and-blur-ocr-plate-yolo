import os
import cv2
import csv
from ultralytics import YOLO

def load_image_paths(directory_path):
    """
    Load image paths from a specified directory.

    Args:
    - directory_path (str): Path to the directory containing images.

    Returns:
    - List[str]: List of file paths to images in the directory.
    """
    return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.jpg') or f.endswith('.png')]

def save_csv(csv_file_path, image_paths, results_dict):
    """
    Save detection results to a CSV file.

    Args:
    - csv_file_path (str): Path to the CSV file.
    - image_paths (List[str]): List of image file paths.
    - results_dict (dict): Dictionary with image paths as keys and detection results as values.
    """
    # Check if the CSV file exists; if not, create a new CSV file and write headers
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image name', 'Detection', 'Score'])  # Column headers

    # Save information to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            detection_count = len(results_dict[image_path]) if results_dict[image_path] != 'None' else 'None'
            scores = ', '.join([f"{score:.2f}" for score in results_dict[image_path]]) if detection_count != 'None' else 'None'
            writer.writerow([filename, detection_count, scores])

def process_images(directory_path, model_path, output_dirs):
    """
    Process images in a directory, performing detection, cropping, and blurring.

    Args:
    - directory_path (str): Path to the directory containing images.
    - model_path (str): Path to the YOLO model.
    - output_dirs (dict): Dictionary containing output directories for blurred images, cropped images, and CSV file.
    """
    # Load image paths
    image_paths = load_image_paths(directory_path)

    # Load YOLO model
    model = YOLO(model_path)  # Using the path to the pre-trained model

    # Create output directories if they don't exist
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Path to the CSV file
    csv_file_path = os.path.join(output_dirs['csv'], 'detections.csv')

    # Dictionary to track detection counts and scores for each file
    results_dict = {}

    # Process all images in the directory
    for image_path in image_paths:
        image = cv2.imread(image_path)

        # Perform detection on the image
        results = model(image)

        detected = False  # Flag indicating if any plates were detected
        confidence_scores = []  # List to store confidence scores

        # Process detection results
        for r in results:
            if hasattr(r, 'boxes'):
                boxes = r.boxes
                for b in boxes:
                    xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                    confidence = float(b.conf)
                    class_id = int(b.cls)
                    class_name = r.names[class_id]
                    label = f"{class_name} {confidence:.2f}"

                    if confidence > 0.35:
                        detected = True
                        confidence_scores.append(confidence)

                        # Draw red bounding box and put confidence score
                        cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                        cv2.putText(image, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        cropped_image = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                        cropped_name = f'{os.path.splitext(os.path.basename(image_path))[0]}.jpg'
                        cv2.imwrite(os.path.join(output_dirs['cropped'], cropped_name), cropped_image)

                        cropped_region = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                        blurred_region = cv2.blur(cropped_region, (23, 23))
                        image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = blurred_region

        # Save modified image to the appropriate folder
        if detected:
            final_image_path = os.path.join(output_dirs['blurred'], f'{os.path.splitext(os.path.basename(image_path))[0]}_Blur.jpg')
            cv2.imwrite(final_image_path, image)
        else:
            final_image_path = os.path.join(output_dirs['blurred'], os.path.basename(image_path))
            cv2.imwrite(final_image_path, image)
            print(f"No license plates were detected in {image_path}.")

        # Save detection results in the dictionary
        results_dict[image_path] = confidence_scores if detected else 'None'

    # Save results to the CSV file
    save_csv(csv_file_path, image_paths, results_dict)

if __name__ == "__main__":
    directory_path = '00_input-blur_B'
    model_path = 'trained_model/train23/weights/90_best.pt'

    # Output directories for saving files
    output_dirs = {
        'blurred': '03_output_blur',
        'cropped': '03_output_cropped',
        'csv': '03_output_csv'
    }

    process_images(directory_path, model_path, output_dirs)
