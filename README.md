# Coin Detection and Image Stitching

## Overview
This project contains two parts:

1. **Part 1: Coin Detection and Segmentation** - Identifies and processes coins in an image using edge detection, Hough Circle Transform, and Watershed segmentation.
2. **Part 2: Image Stitching** - Stitches multiple images together using feature matching and homography transformation.

## Dependencies
Ensure you have the following dependencies installed before running the scripts:

```bash
pip install opencv-python numpy matplotlib scipy scikit-image
```

## Part 1: Coin Detection and Segmentation

### Description
This script processes an image containing coins to:
- Perform edge detection using CLAHE, Gaussian blur, and Canny edge detection.
- Detect and count coins using the Hough Circle Transform.
- Segment coins using morphological operations and the Watershed algorithm.
- Extract individual coins from the image.

### Usage
Ensure you have an input image located at `./input/Part_1/coins.jpeg` and then run:

```bash
python Part_1.py
```
![Uploading 1.jpg…]()

### Outputs
Processed images are saved in `./output/Part_1/`, including:
- `canny_edges.png`: Image with detected edges.
- `count_coins.png`: Image with detected circles.
- `segmented_image.png`: Image with segmented coins.
- `extracted_coins.png`: Extracted individual coins.
- `coin_1.png`, `coin_2.png`, ...: Individual extracted coin images.

## Part 2: Image Stitching

### Description
This script stitches multiple images into a panorama by:
- Detecting keypoints and descriptors using SIFT.
- Matching features between images using Lowe's ratio test.
- Computing homography transformations for alignment.
- Merging images with smooth blending.
- Cropping black regions to finalize the panorama.

### Usage
Ensure input images (`1.jpg`, `2.jpg`, `3.jpg`) are placed in the working directory. Then, run:

```bash
python Part_2.py
```

### Outputs
The final stitched panorama is saved in `./output/Part_2/panorama_output.jpg`.

## Installation and Running the Project

### Clone the Repository
```bash
git clone https://github.com/your-username/coin-detection-image-stitching.git
cd coin-detection-image-stitching
```

### Run the Scripts
```bash
python Part_1.py
python Part_2.py
```

