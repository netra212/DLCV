# Apply different Image Normalization technique from scratch and Visualize the effect. 
# - Do normalization -- see effect on min, max, mean, std value. 
# - See changes in pixel Distribution.
# - 

"""
# Some of the Common Image Normalization Techniques are:- 
1. Min-Max Scaling:
    Description: Rescales the pixel values of the image to a specified range, 
    Typically [0, 1] or [-1, 1].
    X_norm = (X - X_min) / (X_max - X_min)
    where, X_min and X_max are the minimum and maximum pixel values, respectively.
    Use Case: Commonly used when pixel values are in different ranges, and you want to scale them uniformly.

2. Mean Normalization (Zero-Centering):

3. Z-Score Normalization (Standardization):
4. Unit Norm Normalization (L2 Normalization):
5. Global Contrast Normalization (GCN):
6. Histogram Equalization:
7. Adaptive Histogram Equalization (AHE) / Contrast Limited Adaptive Histogram Equalization (CLAHE):
8. Image Whitening (Zero-phase Component Analysis, ZCA):
9. Log Transformation:

10. Gamma Correction:
  
11. Divisive Normalization:
    Description: Normalizes the input across the current batch, ensuring each batch has a mean of 0 and variance of 1.
    Use Case: Used within layers of neural networks to stabilize and accelerate training.

12. Instance Normalization:
    Description: Similar to batch normalization but normalization is performed per instance (image) instead of across the batch.
    Use Case: Commonly used in generative models, like style transfer networks.
"""