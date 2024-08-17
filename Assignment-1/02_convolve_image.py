# Create a function to convolve_image taking image, kernel as input and returned convolved image. 
# Hint: Easy Pesy - done in the class. 

import numpy as np 
from PIL import Image

image_path = "/Users/netrakc/Desktop/DLCV/Assignment-1/amisha3.jpeg"
image = Image.open(image_path)
image_to_numpy_array = np.array(image)

# print(image_to_numpy_array)

# Let's initialize the kernel. 
kernel = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]])

def convolved_image(image_to_numpy_array, kernel):

    """
    The function is to create a convlve image which takes an image, and kernel as input then returned the convolved image. 
    In order to perform that, 
    First we need to load the image, and conver the image into the numpy array. so that we can performs the matrix multiplication with kernels. 
    After that, we have to initialize the kernels. 
    Also, Ensure that shape of the input must be equal to shape of the output or convolved image. 
    """

    input_image_size = image_to_numpy_array.shape
    image_height, image_width = input_image_size
    kernel_height, kernel_width = kernel.shape

    # Dimensional of Output image can be calculated with this formula : (n-k+1)*(n-k+1)
    output_image_height = (image_height - kernel_height + 1)
    output_image_width = (image_width - kernel_width + 1)

    # First Initialize the convolve image with zeros. 
    convolve_image = np.zeros(output_image_height, output_image_width)
    
    # Perform the Convolution Operation.
    for i in range(output_image_height):
        for j in range(output_image_width):
            # Extract the region of the interest to perform the dot product. 
            region_interest = image_to_numpy_array[i:i + kernel_height, j:j + kernel_width]

            # Performs the element-wise Matrix Multiplication and sum the result. 
            convolve_image[i,j] = np.sum(region_interest * kernel)

    return convolve_image


convolved_image(image_to_numpy_array, kernel)

print(image_to_numpy_array.shape)
