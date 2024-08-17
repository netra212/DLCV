# Apply Below Kernels - Note the effects you observe
"""
1. 
[[0, 0, 0],
[0, 1, 0],
[0, 0, 0]]

2. 
[[-1, -1, -1],
[-1,  8, -1],
[-1, -1, -1]]

3. 
[[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]]

4. 
[[-1, -2, -1],
[0,  0,  0],
[1,  2,  1]]

5. 
[[ 0, -1,  0],
[-1,  5, -1],
[ 0, -1,  0]]

6. 
[[-1, -1, -1],
[-1,  8, -1],
[-1, -1, -1]]

7. 
[[-2, -1,  0],
[-1,  1,  1],
[ 0,  1,  2]]

8. 
[[0.111,    0.111,    0.111]
[0.111,    0.111,    0.111]
[0.111,    0.111,    0.111]]
"""
from PIL import Image 
import numpy as np 

image_path = ""
image = Image.open(image_path)
# Converting the image into an numpy array. 
input_image = np.array(image)

identity_kernel = np.array([[0, 0, 0],
                            [0, 1, 0], 
                            [0, 0, 0]])


# Laplacian-kernel is a second order derivative operator used in image preprocessing to detect edges.  Laplacian-kernel is a commonly used discrete approximation of the Laplacian operator. The central value (8) is surrounded by negative values (-1), which sums up the differences with its neighbors, emphasizing edges where intensity changes rapidly. In summary, this is the Laplacian kernel used for edge detection in image processing.
laplacian_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

# Sobel-Operator: This is the Sobel kernel used for detecting horizontal edges in an image. It emphasizes changes in intensity in the horizontal direction.
sobel_kernel_horizontal = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

# Sobel Operator (Vertical) : This is the Sobel kernel used for detecting vertical edges in an image. It emphasizes changes in intensity in the vertical direction.
sobel_kernel_vertical = np.array([[-1, -2, -1],
                                  [0,  0,  0],
                                  [1,  2,  1]])

def apply_identity_kernel(image, kernel):

    # since the shape of the input image must be equal to the shape of the output image. 
    image_height, image_width = image.shape 
    identity_kernel_height,identity_kernel_width = kernel.shape

    # 
    laplacian_kernel_height, laplacian_kernel_width = laplacian_kernel.shape
    sobel_kernel_height, sobel_kernel_width = sobel_kernel_horizontal.shape
    sobel_kernel_vertical_height, sobel_kernel_vertical_width = sobel_kernel_vertical.shape

    output_image_height, output_image_width = (image_height - identity_kernel_height + 1, image_width - identity_kernel_width + 1)

    output_identity_kernel_result = np.zeros(output_image_height, output_image_width)

    for i in range(output_image_height):
        for j in range(output_image_width):
            # Extract the region. 
            region_interest = image[i:i + output_image_height, j:j + output_image_width]
            # sum. 
            output_identity_kernel_result[i,j] = np.sum(region_interest * identity_kernel )

    return output_identity_kernel_result

apply_identity_kernel(image, identity_kernel)