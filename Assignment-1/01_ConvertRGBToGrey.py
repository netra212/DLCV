"""
# Create a function to convert RGB to Grey scale image. Use numpy, pytorch, or tf but don't direct available functions.
Hint: Use the formula: Gray = 0.229R + 0.587G + 0.114*B For the Conversion. 
"""

import numpy as np 
from PIL import Image 

"""
Before performing anything on the input, we need to convert the input image into the numpy array. 
"""

# Loading the image. 
image_path = "/Users/netrakc/Desktop/DLCV/Assignment-1/amisha3.jpeg" 
# Reading the image. 
image = Image.open(image_path)

# Convert the image to a NumPy Array. 
image_np = np.array(image)

print("Shape of the image : ", image_np.shape)

def rgb_to_grey(image_np):

    """
    Approach: 
    - Takes RGB image as input. 
    - Extract the individual Red, Green, and Blue channels. 
    """
    R = image_np[:,:,0]
    G = image_np[:,:,1]
    B = image_np[:,:,2]

    # Now, apply the grayscale conversion. 
    grayscale = 0.229 * R + 0.587 * G + 0.114 * B

    # return the grayscale image. 
    return grayscale.astype(np.uint8)

print(rgb_to_grey(image_np))
