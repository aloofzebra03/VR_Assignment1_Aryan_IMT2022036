import numpy as np
import cv2
import matplotlib.pyplot as plt

# Adjust font size for better visualization
plt.rcParams.update({'font.size': 20})


def compute_perspective_transform(pts1, pts2, correspondences):
    if len(correspondences) < 4:
        raise ValueError("Insufficient keypoint matches!")
    
    # Extract matched keypoints from both images
    src_coords = np.float32([pts1[m.queryIdx].pt for m in correspondences]).reshape(-1, 1, 2)
    dst_coords = np.float32([pts2[m.trainIdx].pt for m in correspondences]).reshape(-1, 1, 2)
    
    # Compute the homography matrix
    transform_matrix, status = cv2.findHomography(src_coords, dst_coords, cv2.RANSAC, 5.0)
    return transform_matrix, status


def crop_black_regions(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    non_black_pixels = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(non_black_pixels)
    return img[y:y+h, x:x+w]


def merge_images(img1, img2, homography):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Define the boundary of the first image
    corners_orig = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    # Transform the boundary points into the second image's space
    corners_transformed = cv2.perspectiveTransform(corners_orig, homography)
    combined_corners = np.concatenate((corners_orig, corners_transformed), axis=0)

    # Compute the size of the stitched image
    [x_min, y_min] = np.int32(combined_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(combined_corners.max(axis=0).ravel() + 0.5)
    shift_vector = [-x_min, -y_min]
    shift_matrix = np.array([[1, 0, shift_vector[0]],
                             [0, 1, shift_vector[1]],
                             [0, 0, 1]], dtype=np.float32)

    # Warp images into the new coordinate space
    transformed_img1 = cv2.warpPerspective(img1, shift_matrix.dot(homography), (x_max - x_min, y_max - y_min))
    transformed_img2 = cv2.warpPerspective(img2, shift_matrix, (x_max - x_min, y_max - y_min))

    # Create masks for smooth blending
    mask1 = np.linspace(1, 0, w1, dtype=np.float32)
    mask2 = np.linspace(0, 1, w2, dtype=np.float32)
    mask1 = np.tile(mask1, (h1, 1))
    mask2 = np.tile(mask2, (h2, 1))

    # Warp masks to match transformed images
    mask1 = cv2.warpPerspective(mask1, shift_matrix.dot(homography), (x_max - x_min, y_max - y_min))
    mask2 = cv2.warpPerspective(mask2, shift_matrix, (x_max - x_min, y_max - y_min))

    # Expand masks to 3-channel format
    mask1 = np.repeat(mask1[:, :, np.newaxis], 3, axis=2)
    mask2 = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)

    # Blend the two images
    final_output = (transformed_img1 * mask1 + transformed_img2 * mask2) / (mask1 + mask2 + 1e-10)
    final_output = np.clip(final_output, 0, 255).astype(np.uint8)
    return final_output


def concatenate_with_spacing(img_list, gap=100):
    h, w, _ = img_list[0].shape
    gap_space = np.zeros((h, gap, 3), dtype=np.uint8)
    gap_space.fill(255)
    result_img = img_list[0]
    for image in img_list[1:]:
        result_img = np.concatenate((result_img, gap_space, image), axis=1)
    return result_img

def detect_and_match_features(img_sequence):
    # Detect SIFT features and generate keypoint visuals
    sift_detector = cv2.SIFT_create()
    keypoints_list, descriptors_list, keypoint_visuals = [], [], []
    for img in img_sequence:
        kp, des = sift_detector.detectAndCompute(img, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
        keypoint_img = cv2.drawKeypoints(img, kp, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoint_visuals.append(keypoint_img)

    # Match features between the first two images
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairwise_matches1 = matcher.knnMatch(descriptors_list[0], descriptors_list[1], k=2)
    pairwise_matches2 = matcher.knnMatch(descriptors_list[1], descriptors_list[2], k=2)

    # Lowe’s ratio test for filtering matches
    filtered_matches1 = [m for m, n in pairwise_matches1 if m.distance < 0.75 * n.distance]
    filtered_matches2 = [m for m, n in pairwise_matches2 if m.distance < 0.75 * n.distance]

    return keypoints_list, descriptors_list, keypoint_visuals, filtered_matches1, filtered_matches2

def visualize_panorama(keypoint_visuals, stitched_final):
    # Convert images from BGR to RGB for matplotlib display
    img1_rgb = cv2.cvtColor(keypoint_visuals[0], cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(keypoint_visuals[1], cv2.COLOR_BGR2RGB)
    img3_rgb = cv2.cvtColor(keypoint_visuals[2], cv2.COLOR_BGR2RGB)
    stitched_rgb = cv2.cvtColor(stitched_final, cv2.COLOR_BGR2RGB)

    # Create a subplot to display the results
    fig, axes = plt.subplots(2, 1, figsize=(15, 10),
                             gridspec_kw={'height_ratios': [1, 2]})
    
    original_with_spacing = concatenate_with_spacing([img1_rgb, img2_rgb, img3_rgb], gap=200)
    axes[0].imshow(original_with_spacing)
    axes[0].set_title("Original Images with Keypoints")
    axes[0].axis("off")

    cv2.imwrite("./output/Part_2/keypoints_detected.jpg", cv2.cvtColor(original_with_spacing, cv2.COLOR_BGR2RGB))


    axes[1].imshow(stitched_rgb)
    axes[1].set_title("Final Stitched Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def create_panorama(img_sequence):
    # Detect and match features between the images
    keypoints_list, descriptors_list, keypoint_visuals, filtered_matches1, filtered_matches2 = detect_and_match_features(img_sequence)
    
    # Compute homography between the first two images and merge
    H1, _ = compute_perspective_transform(keypoints_list[0], keypoints_list[1], filtered_matches1)
    stitched_1 = merge_images(img_sequence[0], img_sequence[1], H1)

    sift_detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # Detect features on the stitched image and match with the third image
    kp_stitched, des_stitched = sift_detector.detectAndCompute(stitched_1, None)
    pairwise_matches_stitched = matcher.knnMatch(des_stitched, descriptors_list[2], k=2)
    filtered_matches_stitched = [m for m, n in pairwise_matches_stitched if m.distance < 0.75 * n.distance]

    # Compute homography between the stitched image and the third image, then merge
    H2, _ = compute_perspective_transform(kp_stitched, keypoints_list[2], filtered_matches_stitched)
    stitched_final = merge_images(stitched_1, img_sequence[2], H2)
    stitched_final = crop_black_regions(stitched_final)

    # Save the final stitched panorama
    cv2.imwrite("./output/Part_2/panorama_output.jpg", stitched_final)


    # Visualize the results
    visualize_panorama(keypoint_visuals, stitched_final)

img_sequence = []
for idx in range(1, 4):
    img = cv2.imread(f"./input/Part_2/{idx}.jpg")
    # img = cv2.imread(f"./{idx}.jpg")
    if img is None:
        raise FileNotFoundError(f"Image {idx}.jpg not found.")
    img = cv2.resize(img, (800, 600))
    img_sequence.append(img)

create_panorama(img_sequence)
