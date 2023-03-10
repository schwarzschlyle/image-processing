# Sample README file

## Contrast Stretching

```python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def contrast_gray(path, lower, upper):
    """
    User defined function to apply contrast stretching to an image
    To increase contrast effects, apply percentile rescaling by clipping pixel values
    
    Parameters:
    
    path (string): set the path of the input image
    lower (int): set the lower percentile clipping lower pixel values
    upper (int): set the higher percentile clipping higher pixel values
    
    Author:
    Lyle Kenneth Geraldez
    """
    
    # Load image
    image = Image.open(path).convert('L')  # Convert to grayscale

    # Calculate minimum and maximum pixel values
    min_val, max_val = np.min(image), np.max(image)

    # Define lower and upper percentile values
    lower_percentile = lower
    upper_percentile = upper

    # Calculate new minimum and maximum pixel values
    new_min_val = np.percentile(image, lower_percentile)
    new_max_val = np.percentile(image, upper_percentile)

    # Apply contrast stretching
    output_image = (image - new_min_val) * (255 / (new_max_val - new_min_val))
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    # Plot histogram before and after contrast stretching
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(np.array(image).ravel(), bins=256, range=(0, 256))
    axs[0].set_title('Histogram before contrast stretching')
    axs[0].set_xlabel('Pixel intensity')
    axs[0].set_ylabel('Frequency')
    axs[1].hist(output_image.ravel(), bins=256, range=(0, 256))
    axs[1].set_title('Histogram after contrast stretching')
    axs[1].set_xlabel('Pixel intensity')
    axs[1].set_ylabel('Frequency')
    plt.savefig(f'contrast_gray_hist_{lower}.jpg', facecolor='black')


    # Display input and output images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image, cmap = 'gray')
    axs[0].set_title('Input image')
    axs[1].imshow(output_image, cmap = 'gray')
    axs[1].set_title('Output image')
    plt.savefig(f'contrast_gray_{lower}.jpg', facecolor='black')
    plt.show()
    
    
for i in range(1, 10, 1):
    contrast_gray(i,100-i)
```