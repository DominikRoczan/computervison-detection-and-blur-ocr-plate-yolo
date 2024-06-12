<h1 style="text-align: center;">Computer Vision Detection and Blur OCR Plate using YOLO</h1>

Welcome to the `computervison-detection-and-blur-ocr-plate-yolo` repository. This project leverages the YOLO (You Only Look Once) model for detecting and blurring OCR plates in images. It is designed to protect privacy by ensuring that sensitive information on vehicle license plates is obscured.

## **Table of Contents**

- **[Introduction](#introduction)**
- **[Features](#features)**
- **[Installation](#installation)**
- **[Usage](#usage)**
- **[Project Structure](#project-structure)**
- **[Contributing](#contributing)**
- **[License](#license)**

## **Introduction**

This repository contains a computer vision application that detects vehicle license plates using the YOLO model and applies a blurring effect to obscure the detected plates. This can be useful for maintaining privacy in images and videos where license plates are visible.

## **Features**

- **License Plate Detection**: Uses YOLO model to detect license plates in images.
- **Blur Functionality**: Applies a blurring effect to the detected license plates to obscure them.
- **High Accuracy**: Utilizes state-of-the-art YOLO model for robust detection.

## **Installation**

To set up this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/DominikRoczan/computervison-detection-and-blur-ocr-plate-yolo.git
    cd computervison-detection-and-blur-ocr-plate-yolo
    ```

2. **Create and activate a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## **Usage**

To use the application, follow these steps:

1. **Prepare your images**: Place the images you want to process in the `input` directory.

2. **Run the detection and blurring script**:
    ```bash
    python blur_license_plates.py
    ```

3. **Output**: The processed images with blurred license plates will be saved in the `output` directory.

## **Project Structure**

