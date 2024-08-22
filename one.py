import cv2
import numpy as np
from window import detect_and_crop_window

# Image Capture and Preprocessing
image_path = '/Users/himanshusharma/Desktop/folders/Project I/images/a2.jpg'
image_with_vehicle = cv2.imread(image_path)
image_without_vehicle = cv2.imread('/Users/himanshusharma/Desktop/folders/Project I/images/b.jpg')

# Convert to grayscale
gray_with_vehicle = cv2.cvtColor(image_with_vehicle, cv2.COLOR_BGR2GRAY)
gray_without_vehicle = cv2.cvtColor(image_without_vehicle, cv2.COLOR_BGR2GRAY)

# Find contours and bounding box for cropping
contours = detect_and_crop_window(image_path)
image_height, image_width = gray_with_vehicle.shape[:2]
image_area = image_height * image_width

window_contour = None
for contour in contours:
    contour_area = cv2.contourArea(contour)
    per = (contour_area / image_area) * 100

    if 1 < per < 5:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 1 < aspect_ratio < 5:
            window_contour = (approx, x, y, w, h)
            break

if window_contour is not None:
    approx, x, y, w, h = window_contour

    # Crop the window region from both images
    cropped_with_vehicle = gray_with_vehicle[y:y + h, x:x + w]
    cropped_without_vehicle = gray_without_vehicle[y:y + h, x:x + w]

    # Apply Canny Edge Detection
    edges_with_vehicle = cv2.Canny(cropped_with_vehicle, 50, 150)
    edges_without_vehicle = cv2.Canny(cropped_without_vehicle, 50, 150)

    # Calculate edge densities
    edge_density_with_vehicle = np.sum(edges_with_vehicle > 0) / (w * h)
    edge_density_without_vehicle = np.sum(edges_without_vehicle > 0) / (w * h)

    # Calculate darkening percentage based on edge density reduction
    if edge_density_without_vehicle > 0:  # Avoid division by zero
        darkening_percentage = (1 - (edge_density_with_vehicle / edge_density_without_vehicle)) * 100
    else:
        darkening_percentage = 0

    print(f"The image became {darkening_percentage:.2f}% darker after applying the tint.")

    # Optionally, display the images
    cv2.imshow('Edges Without Tint', edges_without_vehicle)
    cv2.imshow('Edges With Tint', edges_with_vehicle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
