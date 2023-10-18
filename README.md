# Rotten Apple Detection

This Python script uses computer vision and image processing techniques to detect rotten apples in an image. It applies various image processing methods and analyzes the color distribution to determine the apple's condition.

## Getting Started

Follow the instructions below to get started with the Rotten Apple Detection script.

### Prerequisites

Before you begin, make sure you have the following requirements installed on your system:

- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- ColorThief (optional)

You can install these dependencies using pip:

    pip install opencv-python numpy matplotlib colorthief


### Usage

1. Clone this repository to your local machine:

       git clone https://github.com/yourusername/rotten-apple-detection.git

2. Run the script:

       python rotten_apple_detection.py

 Provide the path to an image of an apple you want to analyze. The script will display the original image, apply various image processing techniques, and determine if the apple is rotten or healthy.

Results

The script will provide the following information:

1. Number of white pixels (representing apple skin)

2. Number of black pixels (representing rotten parts)

3. Percentage of white pixels

4. Percentage of black pixels

If the percentage of black pixels is greater than 5% (you can adjust this threshold), it will label the apple as "Rotten." Otherwise, it will label it as "Healthy."
