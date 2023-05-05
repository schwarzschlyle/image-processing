
# Fourier Rendering Script
# By Schwarzschlyle the Master



# This py script compiles and cleans up the functions used on the jupyter notebook fourier_transform_imaging.ipynb


# Goal: Compile all codes into one function replicating the constructed animation with parameter/s:
#     a) image path
#     and returning the output image. First step would be to save the image frames
#     then, avoiding the intermeddiate step and directly saving rendered GIF




# Status: compiled all draft code snippets




# ---------------------------Images Generation and Saving--------------------------- #


import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os






def fourier_transform(image_path, inner_radius, outer_radius):
    dummy_outer_radius = 500 - outer_radius
    
    # Load image
    img = cv2.imread(image_path, 0)

    # Perform Fourier transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    phase_spectrum = np.angle(fshift)

    # Create mask for circle clipping
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    mask = np.zeros((rows, cols))
    mask_outer = (X - ccol)**2 + (Y - crow)**2 <= outer_radius**2
    mask_inner = (X - ccol)**2 + (Y - crow)**2 <= inner_radius**2
    mask[mask_outer] = 1
    mask[mask_inner] = 0

    # Apply mask to Fourier transform
    fshift_clipped = fshift * mask
    magnitude_spectrum_clipped = 20*np.log(np.abs(fshift_clipped))
    magnitude_spectrum_clipped = np.where(mask != 1, 0, magnitude_spectrum_clipped) # Set clipped area to black
    phase_spectrum_clipped = np.angle(fshift_clipped)

    # Perform inverse Fourier transform on clipped Fourier transform
    f_ishift_clipped = np.fft.ifftshift(fshift_clipped)
    img_clipped = np.fft.ifft2(f_ishift_clipped).real

    
    # Save images
    # plt.imsave('input_image.jpg', img, cmap='gray')
    #plt.imsave(f'magnitude_spectrum_{inner_radius}_{outer_radius}.jpg', magnitude_spectrum, cmap='gray')
    #plt.imsave(f'phase_spectrum_{inner_radius}_{outer_radius}.jpg', phase_spectrum, cmap='gray')
    
    
#      # for a rendering
#     plt.imsave(f'outwards/clipped_img/a_clipped_image_{outer_radius}.jpg', img_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_magnitude/a_clipped_magnitude_spectrum_{outer_radius}.jpg', 
#                magnitude_spectrum_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_phase/a_clipped_phase_spectrum_{outer_radius}.jpg',
#                phase_spectrum_clipped, cmap='gray')
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
#     ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_magnitude/a_3d_clipped_magnitude_spectrum_{outer_radius}.jpg', dpi = 200)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_phase/a_3d_clipped_phase_spectrum_{outer_radius}.jpg', dpi = 200)


    
#     # for b rendering
    
#     plt.imsave(f'outwards/clipped_img/b_clipped_image_{inner_radius}.jpg', img_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_magnitude/b_clipped_magnitude_spectrum_{inner_radius}.jpg', 
#                magnitude_spectrum_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_phase/b_clipped_phase_spectrum_{inner_radius}.jpg',
#                phase_spectrum_clipped, cmap='gray')
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
#     ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_magnitude/b_3d_clipped_magnitude_spectrum_{inner_radius}.jpg', dpi = 200)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_phase/b_3d_clipped_phase_spectrum_{inner_radius}.jpg', dpi = 200)




#     # for c rendering
    
#     dummy_inner_radius = 350 - inner_radius
#     plt.imsave(f'outwards/clipped_img/c_clipped_image_{dummy_inner_radius}.jpg', img_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_magnitude/c_clipped_magnitude_spectrum_{dummy_inner_radius}.jpg', 
#                magnitude_spectrum_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_phase/c_clipped_phase_spectrum_{dummy_inner_radius}.jpg',
#                phase_spectrum_clipped, cmap='gray')
    
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
#     ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_magnitude/c_3d_clipped_magnitude_spectrum_{dummy_inner_radius}.jpg', dpi = 200)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_phase/c_3d_clipped_phase_spectrum_{dummy_inner_radius}.jpg', dpi = 200)
    
    
#     # for d rendering
    
#     dummy_outer_radius = 350 - outer_radius
#     plt.imsave(f'outwards/clipped_img/d_clipped_image_{dummy_outer_radius}.jpg', img_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_magnitude/d_clipped_magnitude_spectrum_{dummy_outer_radius}.jpg', 
#                magnitude_spectrum_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_phase/d_clipped_phase_spectrum_{dummy_outer_radius}.jpg',
#                phase_spectrum_clipped, cmap='gray')


#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
#     ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_magnitude/d_3d_clipped_magnitude_spectrum_{dummy_outer_radius}.jpg', dpi = 200)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_phase/d_3d_clipped_phase_spectrum_{dummy_outer_radius}.jpg', dpi = 200)

    
#     # for e rendering
    
#     plt.imsave(f'outwards/clipped_img/e_clipped_image_{outer_radius}.jpg', img_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_magnitude/e_clipped_magnitude_spectrum_{outer_radius}.jpg', 
#                magnitude_spectrum_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_phase/e_clipped_phase_spectrum_{outer_radius}.jpg',
#                phase_spectrum_clipped, cmap='gray')
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
#     ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_magnitude/e_3d_clipped_magnitude_spectrum_{outer_radius}.jpg', dpi = 200)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_phase/e_3d_clipped_phase_spectrum_{outer_radius}.jpg', dpi = 200)
    
    
    
#     # for f rendering
    
#     plt.imsave(f'outwards/clipped_img/f_clipped_image_{dummy_outer_radius}.jpg', img_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_magnitude/f_clipped_magnitude_spectrum_{dummy_outer_radius}.jpg', 
#                magnitude_spectrum_clipped, cmap='gray')
#     plt.imsave(f'outwards/clipped_phase/f_clipped_phase_spectrum_{dummy_outer_radius}.jpg',
#                phase_spectrum_clipped, cmap='gray')    
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
#     ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_magnitude/f_3d_clipped_magnitude_spectrum_{dummy_outer_radius}.jpg', dpi = 200)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
#     fig.savefig(f'outwards/clipped_3d_phase/f_3d_clipped_phase_spectrum_{dummy_outer_radius}.jpg', dpi = 200)


    return img_clipped


# fourier_transform('image.jpg', 100, 350)


# # a generation (outer-outwards)

# for i in range (0,350,5):
#     fourier_transform('image.jpg',0,i)
    
    
# # b generation (inner-outwards)

# for i in range (0,350,5):
#     fourier_transform('image.jpg',i,350)
    

# # c generation (inner-inwards)

# for i in range (0,350,5):
#     fourier_transform('image.jpg',350-i,350)
    

# # d generation (outer-inwards)

# for i in range (0,350,5):
#     fourier_transform('image.jpg',0 ,350-i)
    

# # e generation (shell-outwards)

# for i in range (0,350,5):
#     fourier_transform('image.jpg',0+i ,30 + i)
    
    
# # f generation (shell-inwards)

# for i in range (0,350,5):
#     fourier_transform('image.jpg', 350 - i ,380 - i)







# ---------------------------Fixing filename padding--------------------------- #


import os
import re

directory = r"C:\Users\Lyle\Desktop\Computation and Modelling\fourier-transform\forensics\outwards\clipped_img"  

# replace with your directory path
prefix = "f_clipped_image_"


# get a list of all files in the directory that start with the prefix
files = [f for f in os.listdir(directory) if f.startswith(prefix)]

for filename in files:
    # extract the number and file extension from the filename
    parts = filename.split("_")
    number = int(parts[-1].split(".")[0])
    extension = parts[-1].split(".")[-1]

    # rename the file with the new padded number
    if number >= 100:
        # already has three digits, do nothing
        new_number = str(number)
    elif number >= 10:
        # add one zero
        new_number = f"0{number}"
    else:
        # add two zeroes
        new_number = f"00{number}"
    new_filename = f"{prefix}{new_number}.{extension}"
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    print(f"Renaming {filename} to {new_filename}")






# ---------------------------GIF rendering--------------------------- #



# using a raw string
# directory = r"C:\Users\Lyle\Desktop\Computation and Modelling\image-processing\sine_plots\varying_freq"
directory = r"C:\Users\Lyle\Desktop\Computation and Modelling\fourier-transform\forensics\outwards\clipped_img"


# create a list of image file names in the directory
image_files = os.listdir(directory)

# sort the list of image file names to ensure proper ordering in the GIF
image_files.sort()

# define the output GIF file name
output_file = "clipped_img.gif"

# create a list to store the image file paths
image_paths = []

# iterate over the list of image file names and add the file path to image_paths
for filename in image_files:
    if filename.endswith(".jpg"):
        image_paths.append(os.path.join(directory, filename))

# create the GIF from the list of image file paths
with imageio.get_writer(output_file, mode="I") as writer:
    for image_path in image_paths:
        # read the image in RGBA format
        image = imageio.imread(image_path, pilmode="RGBA")
        # set the alpha channel to 255 (fully opaque)
        alpha = image[:, :, 3]
        alpha[alpha != 0] = 255
        image[:, :, 3] = alpha
        # append the image to the GIF
        writer.append_data(image)













