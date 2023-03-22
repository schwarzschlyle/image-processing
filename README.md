

To do (Status: 0/6)





> 1. Summary of digital image properties (grayscale, color channels, histograms, CDF) and synthesis
> 2. Summary of color curve manipulation
> 3. Summary of CDF manipulation
> 4. Summary of contrast stretching
> 5. Summary of fade restoration techniques
> 6. Cleanup Fourier filtration rendering function




![](https://i.imgur.com/uucRH8l.png)



# Digital Image Formation and Enhancement

## Contrast Stretching

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def contrast_rgb(path, lower, upper): 
    """
    User defined function to apply contrast stretching to an image
    To increase contrast effects, apply percentile rescaling by clipping pixel values
    For RGB contrasting, we avoid grayscaling the input image
    Histogram is plotted for each color channel
    
    Parameters:
    
    path (string): set the path of the input image
    lower (int): set the lower percentile clipping lower pixel values
    upper (int): set the higher percentile clipping higher pixel values
    
    Author:
    Lyle Kenneth Geraldez
    """
    
    # Load image
    image = Image.open(path)

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
    for i in range(3):
        axs[1].hist(output_image[:,:,i].ravel(), 
        bins=256, range=(0, 256), alpha=0.5, color=['red', 'green', 'blue'][i])
    # axs[1].hist(output_image.ravel(), bins=256, range=(0, 256))
    axs[1].set_title('Histogram after contrast stretching')
    axs[1].set_xlabel('Pixel intensity')
    axs[1].set_ylabel('Frequency')
    # plt.savefig(f'contrast_rgb_hist_{lower}.png', facecolor='black')

    # Display input and output images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input image')
    axs[1].imshow(output_image, cmap='gray')
    axs[1].set_title('Output image')
    # plt.savefig(f'contrast_rgb_{lower}.png', facecolor='black')
    plt.show()

for i in range(1, 10, 1):
    contrast_rgb('dark_galaxy.jpg', i,100-i)
```

```python
contrast_rgb(path, lower, upper)
```


![](https://i.imgur.com/Kl4iMMf.gif)



# Fourier Analysis


## Fourier Filtering

```python
render_fourier_filter()
```




![](https://i.imgur.com/mhmoXc3.gif)
