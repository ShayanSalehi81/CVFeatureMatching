# Image Matching and Keypoint Detection

This repository provides tools for image keypoint detection, feature matching, and homography and fundamental matrix calculation. It includes a Streamlit web app interface for visualizing keypoints and comparing detector performances, as well as core modules for deep feature extraction, feature matching, and image alignment methods like fundamental and homography transformations.

## Project Structure

- **.streamlit/**: Streamlit configuration directory for setting up the web app.
- **app_keypoints.py**: Main application script for Streamlit that allows users to upload images, choose different keypoint detectors, and visualize the keypoints along with processing time statistics.
- **app_matching.py**: Streamlit app for visualizing feature matching between two images.
- **app_reconstruction.py**: Module for reconstructing 3D information based on matched image pairs.
- **deep.py**: Implements deep learning-based feature extractors (e.g., LoFTR and SuperMatcher) using Kornia and custom feature extraction from hloc library.
- **demo.ipynb**: A Jupyter notebook for demonstrating feature matching and keypoint detection functionality.
- **detector.py**: Contains classes for various feature detectors, such as SIFT, ORB, Harris, and more, enabling modular keypoint detection.
- **fundamental.py**: Classes to calculate the fundamental matrix using different methods like RANSAC, LMEDS, and USAC variants.
- **homography.py**: Classes to calculate the homography matrix with methods like RANSAC, LMEDS, and USAC, supporting robust image alignment.
- **matcher.py**: Implements feature matching techniques, including Brute-Force (BF) and FLANN matching, with match filtering functionality.
- **requirements.txt**: List of dependencies required to run the project.
- **utils.py**: Utility functions for image loading, color conversions, and keypoint visualization.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/ShayanSalehi81/CVFeatureMatching
   cd your-repo
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. If using GPU-accelerated features, ensure PyTorch is installed with CUDA support.

## Streamlit Application

The repository includes a Streamlit app for interactive keypoint detection and feature matching. To run the Streamlit app:

```bash
streamlit run app_keypoints.py
```

### Key Features of the Streamlit App

1. **Image Upload and Detection**: Upload an image and detect keypoints using various detectors (SIFT, ORB, Harris, etc.).
2. **Performance Metrics**: Track the processing time and keypoint count for each detection run.
3. **Comparison and Visualization**: Visualize keypoints on the uploaded image, and view charts comparing the performance of different detectors.

## Modules Overview

### Detector Classes (`detector.py`)

Provides implementations for various keypoint detection algorithms, including:

- **SIFTDetector**: Scale-Invariant Feature Transform for robust keypoint detection.
- **ORBDetector**: Efficient alternative to SIFT, optimized for speed.
- **HarrisDetector**: Corner detector based on the Harris matrix.
- **ShiTomasiDetector**: Improved version of Harris for better quality corners.
- Additional detectors: FAST, BRIEF, MSER, AKAZE, BRISK.

Each class has a `detect` method that accepts a grayscale image and returns detected keypoints.

### Matching Classes (`matcher.py`)

Implements two feature-matching strategies:

- **BFMatcher**: Brute-force matcher for direct descriptor matching.
- **FLANNMatcher**: Approximate nearest neighbor matcher, optimized for large datasets.

Both classes support `knnMatch` and `filter_matches` methods to find and filter matches based on a distance ratio.

### Fundamental Matrix (`fundamental.py`)

Classes to compute the fundamental matrix, with support for different methods:

- **DefaultFundamental**: 8-point algorithm.
- **RANSACFundamental**: RANSAC-based method for robustness to outliers.
- **USACMAGSACFundamental**: Advanced USAC variant for better accuracy.
- Additional variants: LMEDS, FM_7POINT, and USAC (DEFAULT, PARALLEL, FAST, ACCURATE).

### Homography Matrix (`homography.py`)

Classes to compute the homography matrix with support for RANSAC, LMEDS, and various USAC methods, enabling robust image alignment for tasks like panorama stitching.

### Deep Learning-Based Matchers (`deep.py`)

Integrates deep learning-based feature extraction and matching methods using Kornia and the hloc library:

- **LoFTRMatcher**: Uses the LoFTR model for dense feature matching, capable of capturing long-range dependencies in challenging scenes.
- **SuperMatcher**: Combines feature extraction and matching for high-performance applications like 3D reconstruction.

### Utilities (`utils.py`)

Helper functions for image handling and visualization:

- **load_image_from_bytes**: Reads images from byte input for Streamlit upload handling.
- **to_gray** and **to_rgb**: Color conversion functions.
- **draw_keypoints** and **draw_loftr_keypoints**: Functions to overlay detected keypoints on images for visualization.

## Demonstrations and Usage

### Keypoint Detection Example

Here's an example of how to use the `SIFTDetector` class:

```python
from detector import SIFTDetector
import cv2

# Load and convert image to grayscale
image = cv2.imread('image.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect keypoints
detector = SIFTDetector()
keypoints = detector.detect(gray_image)

# Draw and display keypoints
output_image = draw_keypoints(image, keypoints)
cv2.imshow("Keypoints", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Feature Matching Example

Using `BFMatcher` for feature matching:

```python
from matcher import BFMatcher
from detector import ORBDetector

detector = ORBDetector()
matcher = BFMatcher()

# Detect keypoints and descriptors for two images
keypoints1, descriptors1 = detector.detect_and_compute(img1)
keypoints2, descriptors2 = detector.detect_and_compute(img2)

# Match descriptors
matches = matcher.knnMatch(descriptors1, descriptors2)

# Filter matches
good_matches = matcher.filter_matches(matches)
```

### Fundamental and Homography Estimation

Estimating fundamental and homography matrices between two sets of points:

```python
from fundamental import RANSACFundamental
from homography import RANSACHomography

# Define matched points (src_pts, dst_pts)
fundamental_matrix = RANSACFundamental().findFundamental(src_pts, dst_pts)
homography_matrix = RANSACHomography().findHomography(src_pts, dst_pts)
```

## Future Improvements

- **Enhanced Deep Learning Models**: Integrate additional feature extractors for improved performance in challenging scenes.
- **3D Reconstruction**: Add more comprehensive tools for 3D point cloud reconstruction.
- **Web Interface**: Expand the Streamlit app to support feature matching and transformation estimations in a more interactive format.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you'd like to add features or fix bugs.
