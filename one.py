import cv2
import numpy as np
import matplotlib.pyplot as plt
from window import detect_and_crop_window

# Image Capture and Preprocessing
image_path = '/Users/himanshusharma/Desktop/folders/Project I/images/a1.jpg'
image_with_vehicle = cv2.imread('/Users/himanshusharma/Desktop/folders/Project I/images/a2.jpg', cv2.IMREAD_GRAYSCALE)
image_without_vehicle = cv2.imread('/Users/himanshusharma/Desktop/folders/Project I/images/b.jpg', cv2.IMREAD_GRAYSCALE)


# Find contours and bounding box for cropping
contours = detect_and_crop_window(image_path)

# Assuming the largest contour is the region of interest (ROI)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the region of interest (upper half of the bounding box, assuming it represents the window)
    window_crop_with_vehicle = image_with_vehicle[y:y + int(h * 0.5), x:x + w]
    window_crop_without_vehicle = image_without_vehicle[y:y + int(h * 0.5), x:x + w]

    # Calculate the mean intensity of both images
    mean_intensity_original = np.mean(window_crop_without_vehicle)
    mean_intensity_screened = np.mean(window_crop_with_vehicle)

    # Calculate how much darker the screened image is compared to the original
    if mean_intensity_original > 0:  # Avoid division by zero
        darkening_percentage = (abs(mean_intensity_original - mean_intensity_screened) / mean_intensity_original) * 100
    else:
        darkening_percentage = 0

    # Output the results
    print(f"The image became {darkening_percentage:.2f}% darker after applying the screen.")

    # Optionally, display the images
    cv2.imshow('Original Image', window_crop_without_vehicle)
    cv2.imshow('Screened Image', window_crop_with_vehicle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    