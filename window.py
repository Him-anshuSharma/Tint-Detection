import cv2
import numpy as np

def detect_and_crop_window(image_path, area_threshold=(1, 5), aspect_ratio_threshold=(1, 5)):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path and file integrity.")
        return None

    # Calculate image dimensions and area
    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width

    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find Canny edges
    edged = cv2.Canny(blurred, 30, 200)

    # Finding contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

   