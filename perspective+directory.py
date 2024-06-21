import cv2
import numpy as np
import os

def load_image(path):
    return cv2.imread(path)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    return edged

def find_largest_contour(edged):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def get_box_points(largest_contour):
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    return box.astype('int')

def draw_contour(image, box):
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    return image

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def perspective_transform(image, box):
    rect = order_points(box)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = load_image(image_path)
            if image is None:
                continue
            print(f"Processing {filename}")

            edged = preprocess_image(image)
            largest_contour = find_largest_contour(edged)
            box = get_box_points(largest_contour)

            warped_image = perspective_transform(image, box)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, warped_image)

def main():
    input_folder = '03_output_cropped'
    output_folder = '04_output_cropped_transform'
    process_images(input_folder, output_folder)

if __name__ == "__main__":
    main()
