import os
from PIL import Image

def create_output_directory(output_directory):
    """
    Create the output directory if it does not exist.

    Args:
        output_directory (str): Path to the output directory.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

def convert_webp_to_jpg(input_directory, output_directory):
    """
    Convert all WEBP images in the input directory to JPG format and save them in the output directory.

    Args:
        input_directory (str): Path to the input directory containing WEBP images.
        output_directory (str): Path to the output directory where JPG images will be saved.
    """
    # Ensure the output directory exists
    create_output_directory(output_directory)

    # Process each file in the input directory
    for index, filename in enumerate(sorted(os.listdir(input_directory))):
        if filename.endswith('.webp'):
            # Create the full file path
            file_path = os.path.join(input_directory, filename)

            # Open the WEBP image
            image = Image.open(file_path)

            # Convert the image to JPG format
            rgb_image = image.convert('RGB')

            # Create a new filename with appropriate numbering and original name
            new_filename = '{:02d}_{}.jpg'.format(index + 1, os.path.splitext(filename)[0])
            new_file_path = os.path.join(output_directory, new_filename)

            # Save the image in JPG format
            rgb_image.save(new_file_path, 'JPEG', quality=90)

    print("Conversion completed.")

if __name__ == "__main__":
    # Path to the directory containing WEBP images
    input_directory = '01_input_webp'

    # Path to the directory where JPG images will be saved
    output_directory = '02_input_blur'

    # Convert WEBP images to JPG
    convert_webp_to_jpg(input_directory, output_directory)
