import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import imageio
from skimage.draw import disk
from skimage.draw import rectangle


# Create a binary image of two dots (one pixel each) along the x-axis symmetric about center.
img = np.zeros((64, 64))
img[32, 21] = 1
img[32, 43] = 1

# Compute the Fourier transform of the image.
ft = np.fft.fftshift(np.fft.fft2(img))

# Compute the modulus of the Fourier transform.
modulus = np.abs(ft)

# Display the modulus.
# plt.imshow(img, cmap='gray')
# plt.imshow(modulus, cmap='gray')
plt.imsave('img.jpg',
                           img, cmap='gray')
plt.imsave('fft.jpg',
                     modulus, cmap='gray')




# Create the disk directory
directory = 'disk'
os.makedirs(directory)

# Create the subdirectories
subdirectory1 = os.path.join(directory, 'img')
os.makedirs(subdirectory1)

subdirectory2 = os.path.join(directory, 'fft')
os.makedirs(subdirectory2)
 

# Create the rectangle directory
directory = 'rectangle'
os.makedirs(directory)

# Create the subdirectories
subdirectory1 = os.path.join(directory, 'img')
os.makedirs(subdirectory1)

subdirectory2 = os.path.join(directory, 'fft')
os.makedirs(subdirectory2)
 





def ft_disk(r):
    img = np.zeros((64, 64))
    rr, cc = disk((32, 21), r)
    img[rr, cc] = 1
    rr, cc = disk((32, 43), r)
    img[rr, cc] = 1

    # Compute the Fourier transform of the image.
    ft = np.fft.fftshift(np.fft.fft2(img))

    # Compute the modulus of the Fourier transform.
    modulus = np.abs(ft)

    # Display the modulus.
    #plt.imshow(modulus, cmap='gray')
    # plt.imshow(img, cmap='gray')
    #plt.show()
    
    if r<10:
        plt.imsave(f'disk/img/0{r}.jpg',
                           img, cmap='gray')
        plt.imsave(f'disk/fft/0{r}.jpg',
                           modulus, cmap='gray')
    else:
        plt.imsave(f'disk/img/{r}.jpg',
                           img, cmap='gray')
        plt.imsave(f'disk/fft/{r}.jpg',
                           modulus, cmap='gray')

for i in range(1,21):
    ft_disk(i)





def ft_rectangle(s):
    img = np.zeros((64, 64))
    img[32, 21] = 1
    img[32, 43] = 1

    # Set the side length of the rectangles


    # Replace the points with rectangles
    for r, c in zip(*np.where(img)):
        rr, cc = rectangle((r-s//2, c-s//2), (r+s//2, c+s//2))
        img[rr, cc] = 1

    # Compute the Fourier transform of the image.
    ft = np.fft.fftshift(np.fft.fft2(img))

    # Compute the modulus of the Fourier transform.
    modulus = np.abs(ft)

    # Display the modulus.
    # plt.imshow(img, cmap='gray')
    # plt.imshow(modulus, cmap='gray')
    
    if s<10:
        plt.imsave(f'rectangle/img/0{s}.jpg',
                           img, cmap='gray')
        plt.imsave(f'rectangle/fft/0{s}.jpg',
                           modulus, cmap='gray')
    else:
        plt.imsave(f'rectangle/img/{s}.jpg',
                           img, cmap='gray')
        plt.imsave(f'rectangle/fft/{s}.jpg',
                           modulus, cmap='gray')

for i in range(1,21):
    ft_rectangle(i)




# using a raw string

directory = "rectangle/img"


# create a list of image file names in the directory
image_files = os.listdir(directory)

# sort the list of image file names to ensure proper ordering in the GIF
image_files.sort()

# define the output GIF file name
output_file = "rectangle_img.gif"

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







# using a raw string

directory = "rectangle/fft"


# create a list of image file names in the directory
image_files = os.listdir(directory)

# sort the list of image file names to ensure proper ordering in the GIF
image_files.sort()

# define the output GIF file name
output_file = "rectangle_fft.gif"

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







# using a raw string

directory = "disk/img"


# create a list of image file names in the directory
image_files = os.listdir(directory)

# sort the list of image file names to ensure proper ordering in the GIF
image_files.sort()

# define the output GIF file name
output_file = "disk_img.gif"

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





# using a raw string

directory = "disk/fft"


# create a list of image file names in the directory
image_files = os.listdir(directory)

# sort the list of image file names to ensure proper ordering in the GIF
image_files.sort()

# define the output GIF file name
output_file = "disk_fft.gif"

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
