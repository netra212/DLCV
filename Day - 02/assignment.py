
# Q1. Create a function to convert RGB to Grey scale Image. Use Numpy, PyTorch, or tf but don't use directly avaiable functions. 
import cv2
import numpy 

image_path = "/Users/netrakc/Desktop/DLCV/Day - 02/ai image.jpeg"
# reading an image. 
image = cv2.imread(image_path)

def convertRGB2Grey(image):
    """
    Gray = 0.299R + 0.587G + 0.114*B"""

    height, width, color_channels = image.shape
    print("Height: ", height, "| width: ", width, "| color_channels: ", color_channels)

    # Converting RGB Image to Greyscale. 
    # First need to split the number of channels like this. 
    # [n1, n2, 1], [n1, n2, 1], [n1, n2, 1]

print(convertRGB2Grey(image))

# Q2 : Create a function to convolve_image taking image, kernel as input and returned convolved image.

def convolve_image():
    """"""
    pass

# Q3 : Apply Kernels - Note the effects. 
def apply_kernels():
    """
    a.
    [[0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]]

    b. 
    [[-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]]

    c.
    [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]

    d.
    [[-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]]

    e.
    [[ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]]

    f.
    [[-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]]

    g.
    [[-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]]

    h.
    [[0.111,    0.111,    0.111]
    [0.111,    0.111,    0.111]
    [0.111,    0.111,    0.111]]
    """

# Question 4: Image Filtering and Transformation
def image_filtering_transformation():
    """
    Task: Implement a function to perform image filtering and transformation that includes:

        * Filtering: Apply a Gaussian blur to the image to reduce noise and smooth the image.
        * Transformation: Apply a rotation to the image by a specified angle.
    
    Hints:

        * For Gaussian blur, you can create a Gaussian kernel and apply it using convolution. https://en.wikipedia.org/wiki/Gaussian_blur

        * For rotation, compute the rotation matrix and use cv2.warpAffine. https://theailearner.com/tag/cv2-warpaffine/
    """
    pass

# Question 5: Normalization
def normalization():
    """
    Apply different Image Normalization technique from scratch and visualize the effect.

        * Do normalization - see effect on min, max, mean, std value.
        * See changes in Pixel Distribution.
    """

