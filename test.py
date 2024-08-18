import cv2
import numpy as np

# Load the image
image_path = '/Users/himanshusharma/Desktop/folders/Project I/images/a2.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at {image_path}. Please check the file path and file integrity.")
else:
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
    contours, hierarchy = cv2.findContours(edged.copy(), 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the best contour
    window_contour = None

    for contour in contours:
        # Calculate area of the contour
        contour_area = cv2.contourArea(contour)

        # Calculate the percentage area of the contour relative to the image area
        per = (contour_area / image_area) * 100

        # Filter contours based on area percentage
        if 1 < per < 5:

            # Approximate the contour to reduce the number of points
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)


            # Calculate bounding box and aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Filter based on aspect ratio (typically close to 1.5 - 2.5 for car windows)
            if 1 < aspect_ratio < 5:
                window_contour = (approx, x, y, w, h)
                break  # Exit the loop if a suitable contour is found

    if window_contour is not None:
        approx, x, y, w, h = window_contour

        # Draw the contour of the window on the original image
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 1)

        # Crop the detected window from the image
        cropped_window = image[y:y + h, x:x + w]

        # Save the cropped window image
        cv2.imwrite('/Users/himanshusharma/Desktop/folders/Project I/images/cropped_window.jpg', cropped_window)

        # Display the image with the detected window contour
        cv2.imshow('Window Contour', image)
        cv2.imshow('Cropped Window', cropped_window)
        cv2.waitKey(0)
    else:
        print("No window contour found.")

    cv2.destroyAllWindows()
