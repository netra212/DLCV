import numpy as np
import cv2 

image_path = "/Users/netrakc/Desktop/DLCV/Day - 02/ai image.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

print("Displaying an image: ")
print(image)

# Now, Implementing an sobel_vertical as kernel. 
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

print("sobel vertical: ")
print(sobel_vertical)

# Checking image shape. 
print("Image shape: ")
print(image.shape)

# Fetching rows and columns.
rows, cols = image.shape

# 
k = sobel_vertical.shape[0]
print("Sobel Vertical: ")
print(k)

# Output matrix. 
output_matrix = np.zeros((rows-k + 1, cols-k+1))
print("Output Matrix: \n", output_matrix)

# 
for i in range(1, rows-1):
    for j in range(1, cols-1):
        # Performing Convolutional Operation. 
        region = image[i-1:i+2, j-1:j+2]
        output_matrix[i-1, j-1] = np.sum(region * sobel_vertical)

print("Matrix After Convolutional Operation: ")
print(output_matrix)