import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image Capture and Preprocessing
image_with_vehicle = cv2.imread('/Users/himanshusharma/Desktop/folders/Project I/images/a.jpg', cv2.IMREAD_GRAYSCALE)
image_without_vehicle = cv2.imread('/Users/himanshusharma/Desktop/folders/Project I/images/b.jpg', cv2.IMREAD_GRAYSCALE)

# Intensity Normalization
def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

normalized_image_with_vehicle = normalize_image(image_with_vehicle)
normalized_image_without_vehicle = normalize_image(image_without_vehicle)

# Background Subtraction
difference_image = cv2.absdiff(normalized_image_with_vehicle, normalized_image_without_vehicle)

# Apply Thresholding to isolate changes
_, thresholded_image = cv2.threshold(difference_image, 30, 255, cv2.THRESH_BINARY)

# Morphological Operations (Erosion and Dilation)
kernel = np.ones((5, 5), np.uint8)
morph_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)

# Find contours and bounding box for cropping
contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Assuming the largest contour is the region of interest (ROI)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Improve window cropping by targeting the window area (upper portion of bounding box)
    window_crop_with_vehicle = normalized_image_with_vehicle[y:y + int(h * 0.5), x:x + w]
    window_crop_without_vehicle = normalized_image_without_vehicle[y:y + int(h * 0.5), x:x + w]

    # Calculate the mean intensity difference
    mean_intensity_with_vehicle = np.mean(window_crop_with_vehicle)
    mean_intensity_without_vehicle = np.mean(window_crop_without_vehicle)
    
    # Calculate the tint level based on how much darker the foreground image is
    tint_level = (abs(mean_intensity_without_vehicle - mean_intensity_with_vehicle)) / mean_intensity_without_vehicle

    # Display results
    # cv2.imshow('Original Image with Vehicle', image_with_vehicle)
    cv2.imshow('Normalized Image with Vehicle', normalized_image_with_vehicle)
    # cv2.imshow('Difference Image', difference_image)
    # cv2.imshow('Thresholded Image', thresholded_image)
    cv2.imshow('Morph Image', morph_image)
    # cv2.imshow('Cropped Window Image with Vehicle', window_crop_with_vehicle)
    # cv2.imshow('Cropped Window Image without Vehicle', window_crop_without_vehicle)

    # Save the cropped window images
    cv2.imwrite('cropped_window_image_with_vehicle.jpg', window_crop_with_vehicle)
    cv2.imwrite('cropped_window_image_without_vehicle.jpg', window_crop_without_vehicle)

    # Print Tint Level
    print(f"Mean Intensity with Vehicle: {mean_intensity_with_vehicle}")
    print(f"Mean Intensity without Vehicle: {mean_intensity_without_vehicle}")
    print(f"Tint Level (how much darker): {tint_level:.2f}")

    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found. Image processing could not be completed.")