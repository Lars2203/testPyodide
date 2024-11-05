import numpy as np
from skimage import io, filters, morphology
import matplotlib.pyplot as plt

# Load and display the original grayscale image
image = io.imread('gray.png', as_gray=True)
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Thresholding
threshold_value = filters.threshold_otsu(image)
binary_image = image > threshold_value
plt.subplot(2, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image After Thresholding')
plt.axis('off')

# Dilation
dilated_image = morphology.dilation(binary_image, morphology.disk(3))
plt.subplot(2, 3, 3)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated Image')
plt.axis('off')

# Erosion
eroded_image = morphology.erosion(binary_image, morphology.disk(3))
plt.subplot(2, 3, 4)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')
plt.axis('off')

# Opening
opened_image = morphology.opening(binary_image, morphology.disk(3))
plt.subplot(2, 3, 5)
plt.imshow(opened_image, cmap='gray')
plt.title('Opened Image')
plt.axis('off')

# Closing
closed_image = morphology.closing(binary_image, morphology.disk(3))
plt.subplot(2, 3, 6)
plt.imshow(closed_image, cmap='gray')
plt.title('Closed Image')
plt.axis('off')

plt.tight_layout()
plt.show()
