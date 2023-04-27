


import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import imageio
from PIL import Image
import ast




import warnings
warnings.filterwarnings("ignore")





def clip(image_path, inner_radius, outer_radius, num_masks, mask_origins):
    
    # Load image
    img = cv2.imread(image_path, 0)

    # Mean center
    mean_value = np.mean(img)
    img = np.subtract(img, mean_value)
    

    # Perform Fourier transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    phase_spectrum = np.angle(fshift)

    # Create mask for circle clipping
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    mask = np.ones((rows, cols))


    # Clips a circular mask cener points mask_origins
    for i in range(num_masks):
        mask_outer = (X - mask_origins[i][1])**2 + (Y - mask_origins[i][0])**2 <= outer_radius**2
        mask_inner = (X - mask_origins[i][1])**2 + (Y - mask_origins[i][0])**2 <= inner_radius**2
        mask[mask_outer] = 0
        mask[mask_inner] = 1

        
    # Apply mask to Fourier transform
    fshift_clipped = fshift * mask
    magnitude_spectrum_clipped = 20*np.log(np.abs(fshift_clipped))
    magnitude_spectrum_clipped = np.where(mask != 1, 0, magnitude_spectrum_clipped) # Set clipped area to black
    phase_spectrum_clipped = np.angle(fshift_clipped)

    # Perform inverse Fourier transform on clipped Fourier transform
    f_ishift_clipped = np.fft.ifftshift(fshift_clipped)
    img_clipped = np.fft.ifft2(f_ishift_clipped).real


    
    directory = f'{image_path}_clipping'
    os.makedirs(directory)

    

    plt.imsave(f'{image_path}_clipping/img_clipped.jpg', 
               img_clipped, cmap='gray')
    plt.imsave(f'{image_path}_clipping/magnitude_spectrum.jpg', 
               magnitude_spectrum_clipped, cmap='gray')
    plt.imsave(f'{image_path}_clipping/phase_spectrum.jpg',
               phase_spectrum_clipped, cmap='gray')



    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(magnitude_spectrum.shape[1]), np.arange(magnitude_spectrum.shape[0]))
    ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
    fig.savefig(f'{image_path}_clipping/3d_magnitude_spectrum.jpg', dpi = 200)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
    fig.savefig(f'{image_path}_clipping/3d_phase_spectrum.jpg', dpi = 200)
    
    
    
    
    
if __name__ == "__main__":
    
    
    # get the command-line arguments
    image_path = sys.argv[1]
    inner_radius = sys.argv[2]
    outer_radius = sys.argv[3]
    num_masks = sys.argv[4]
    masks_origin = sys.argv[5]
    
    # call the function with the parameters
    print("Welcome to FourierViz by Schwarzschlyle")
    print("Please wait while I explore the Fourier information of your image. Thank you!")
    print("To gain more information regarding possible clipping regions according to your desired application, you can try using fourier_sweep.py")
    print("Clipping image...")
    
    
    
    clip(image_path, int(inner_radius), int(outer_radius), int(num_masks), ast.literal_eval(masks_origin))
    
    
