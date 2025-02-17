import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = "./output/Part_1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.rcParams.update({'font.size': 20})

def show_image(title, image, cmap=None):
    """Display an image with a title using matplotlib."""
    plt.figure(figsize=(8, 6))
    if cmap != None:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

def find_edges(input_image_path):
    """Find edges in the input image using CLAHE, Gaussian blur, and Canny edge detection."""
    frame = cv2.imread(input_image_path)
    if frame is None:
        print("Could not open image")
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=5.5, tileGridSize=(8, 8))
    clahe_result = clahe.apply(frame_gray)
    
    # Apply Gaussian blur to reduce noise
    blurred_result = cv2.GaussianBlur(clahe_result, (19, 19), 0)
    
    # Detect edges using Canny
    canny_edges = cv2.Canny(blurred_result, 229, 0)
    
    cv2.imwrite(f"{output_dir}/canny_edges.png", canny_edges)
    show_image("Canny Edges", canny_edges, cmap='gray')
    return frame_rgb, blurred_result, canny_edges

def count_coins_hough(image_path):
    """Count the number of coins using Hough Circle Transform and display detected circles."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                param1=100, param2=25, minRadius=75, maxRadius=100)
    
    detected_image = image.copy()
    coin_count = 0
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        coin_count = len(circles[0, :])
        for i in circles[0, :]:
            cv2.circle(detected_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(detected_image, (i[0], i[1]), 2, (0, 0, 255), 3)  # Center marker
    
    detected_display = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{output_dir}/count_coins.png", detected_image)
    show_image(f"Detected Circles ({coin_count})", detected_display)
    print(f"Detected number of coins using Hough Circles: {coin_count}")
    return coin_count

def segment_coins(image_path):
    """Segment coins from the image using morphological operations and watershed algorithm."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (45, 45), 0)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological transformations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=21)
    sure_bg = cv2.dilate(opening, kernel, iterations=100)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    
    # Create segmented output
    segmented_image = np.zeros_like(image)
    for marker in range(2, markers.max() + 1):
        segmented_image[markers == marker] = np.random.randint(0, 255, 3).tolist()
    
    segmented_display = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{output_dir}/segmented_image.png", segmented_display)
    show_image("Segmented Image", segmented_display)
    return segmented_display, markers

def extract_coins(image_path, markers):
    """Extract individual coins from the segmented image, save and display them."""
    image = cv2.imread(image_path)
    extracted = []
    extracted_display = image.copy()
    
    count = 1  # Start naming coins from 1
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.uint8(markers == label) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            coin = image[y:y+h, x:x+w]
            coin_rgb = cv2.cvtColor(coin, cv2.COLOR_BGR2RGB)
            extracted.append(coin_rgb)
            coin_filename = f"{output_dir}/coin_{count}.png"
            cv2.imwrite(coin_filename, coin)
            cv2.rectangle(extracted_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            show_image(f"Extracted Coin {count}", coin_rgb)
            count += 1
    
    cv2.imwrite(f"{output_dir}/extracted_coins.png", extracted_display)
    extracted_display = cv2.cvtColor(extracted_display, cv2.COLOR_BGR2RGB)
    show_image("All Extracted Coins", extracted_display)
    return extracted
# Define input image path
image_path = './input/Part_1/coins.jpeg'

# Run edge detection
frame_rgb, blurred_result, canny_edges = find_edges(image_path)

# Perform segmentation
segmented_image, markers = segment_coins(image_path)

# Extract coins using markers
extracted_coins = extract_coins(image_path, markers)

# Count coins using Hough Circle Transform
coin_count_hough = count_coins_hough(image_path)

print("Processing completed. Images saved in ./output/Part_2")
