import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

# Create the main directory
directory = 'random'
os.makedirs(directory)

# Create the subdirectories
subdirectory1 = os.path.join(directory, 'pattern_img')
os.makedirs(subdirectory1)

subdirectory2 = os.path.join(directory, 'pattern_fft')
os.makedirs(subdirectory2)

subdirectory3 = os.path.join(directory, 'arr_img')
os.makedirs(subdirectory3)

subdirectory4 = os.path.join(directory, 'arr_fft')
os.makedirs(subdirectory4)

subdirectory5 = os.path.join(directory, 'conv_img')
os.makedirs(subdirectory5)

subdirectory6 = os.path.join(directory, 'conv_fft')
os.makedirs(subdirectory6)





# Create the main directory
directory = 'uniform'
os.makedirs(directory)

# Create the subdirectories
subdirectory1 = os.path.join(directory, 'pattern_img')
os.makedirs(subdirectory1)

subdirectory2 = os.path.join(directory, 'pattern_fft')
os.makedirs(subdirectory2)

subdirectory3 = os.path.join(directory, 'arr_img')
os.makedirs(subdirectory3)

subdirectory4 = os.path.join(directory, 'arr_fft')
os.makedirs(subdirectory4)

subdirectory5 = os.path.join(directory, 'conv_img')
os.makedirs(subdirectory5)

subdirectory6 = os.path.join(directory, 'conv_fft')
os.makedirs(subdirectory6)






def generate_random_ones(w):

    # Set the seed for the random number generator
    np.random.seed(w)

    # Create a 200x200 array of zeros
    random_arr = np.zeros((200, 200))

    # Put 10 1's in random locations in the array
    indices = np.random.choice(range(200*200), 10, replace=False)
    random_arr[np.unravel_index(indices, (200,200))] = 1
    
    arr = random_arr
    

    # Create an arbitrary 9x9 pattern
    pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1]])


    # Zero-pad the pattern to 200x200 size
    pattern_padded = np.pad(pattern, ((95,96),(95,96)), mode='constant')


    # Take the FFT of both arrays
    fft_arr = np.fft.fft2(arr)
    fft_pattern = np.fft.fft2(pattern_padded)


    # plt.imshow(pattern, cmap='gray')
    # plt.show()
    if w < 10: 
        plt.imsave(f'random/pattern_img/0{w}.jpg',
                               pattern, cmap='gray')
    else:
        plt.imsave(f'random/pattern_img/{w}.jpg',
                               pattern, cmap='gray')


    # Display the modulus of the FFT of the pattern
    
    
    # plt.imshow(np.abs(np.fft.fftshift(fft_pattern)), cmap='gray')
    # plt.show()
    
    if w < 10:
        plt.imsave(f'random/pattern_fft/0{w}.jpg',
                               np.abs(np.fft.fftshift(fft_pattern)), cmap='gray')
    else:
        plt.imsave(f'random/pattern_fft/{w}.jpg',
                           np.abs(np.fft.fftshift(fft_pattern)), cmap='gray')


    # Multiply the FFTs elementwise
    product_fft = fft_arr * fft_pattern

    # Take the inverse FFT of the product
    product = np.fft.ifft2(product_fft)

    # Apply fftshift to center the result
    product = np.fft.fftshift(product)


    # Display arr
    # plt.imshow(arr, cmap='gray')
    # plt.show()
    
    if w < 10:
        plt.imsave(f'random/arr_img/0{w}.jpg',
                               arr, cmap='gray')
    else:
        plt.imsave(f'random/arr_img/{w}.jpg',
                           arr, cmap='gray')
    

    # Display the modulus of the FFT of the array
    # plt.imshow(np.abs(np.fft.fftshift(fft_arr)), cmap='gray')
    # plt.show()
    
    if w < 10:
        plt.imsave(f'random/arr_fft/0{w}.jpg',
                               np.abs(np.fft.fftshift(fft_arr)), cmap='gray')
    else:
        plt.imsave(f'random/arr_fft/{w}.jpg',
                           np.abs(np.fft.fftshift(fft_arr)), cmap='gray')


    # Display the resulting image
    # plt.imshow(np.abs(product), cmap='gray')
    # plt.show()
    if w < 10:
        plt.imsave(f'random/conv_img/0{w}.jpg',
                               np.abs(product), cmap='gray')
    else:
        plt.imsave(f'random/conv_img/{w}.jpg',
                           np.abs(product), cmap='gray')


    # Display the modulus of the FFT of the pattern
    # plt.imshow(np.abs(np.fft.fftshift(product_fft)), cmap='gray')
    # plt.show()
    
    if w < 10:
        plt.imsave(f'random/conv_fft/0{w}.jpg',
                               np.abs(np.fft.fftshift(product_fft)), cmap='gray')
    else:
        plt.imsave(f'random/conv_fft/{w}.jpg',
                           np.abs(np.fft.fftshift(product_fft)), cmap='gray')
    
    return random_arr




def generate_uniform_interval_ones(w):
    
    # Create a 200x200 array of zeros
    uniform_arr = np.zeros((200, 200))

    # Create a grid of points spaced by w
    x_idx = np.arange(w, 200, w)
    y_idx = np.arange(w, 200, w)
    xx, yy = np.meshgrid(x_idx, y_idx)

    # Set the values at the grid points to 1
    uniform_arr[xx, yy] = 1

    # Add some noise to make it more interesting
    # arr += np.random.rand(200, 200) < 0.1

    # Ensure values are either 0 or 1
    uniform_arr = np.clip(uniform_arr, 0, 1)
    
    # Set the seed for the random number generator
    np.random.seed(123)

    # Create a 200x200 array of zeros
    random_arr = np.zeros((200, 200))

    # Put 10 1's in random locations in the array
    indices = np.random.choice(range(200*200), 10, replace=False)
    random_arr[np.unravel_index(indices, (200,200))] = 1
    
    arr = uniform_arr
    

    # Create an arbitrary 9x9 pattern
    pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1]])


    # Zero-pad the pattern to 200x200 size
    pattern_padded = np.pad(pattern, ((95,96),(95,96)), mode='constant')


    # Take the FFT of both arrays
    fft_arr = np.fft.fft2(arr)
    fft_pattern = np.fft.fft2(pattern_padded)


    if w < 10: 
        plt.imsave(f'uniform/pattern_img/0{w}.jpg',
                               pattern, cmap='gray')
    else:
        plt.imsave(f'uniform/pattern_img/{w}.jpg',
                               pattern, cmap='gray')


    # Display the modulus of the FFT of the pattern
    
    
    # plt.imshow(np.abs(np.fft.fftshift(fft_pattern)), cmap='gray')
    # plt.show()
    
    if w < 10:
        plt.imsave(f'uniform/pattern_fft/0{w}.jpg',
                               np.abs(np.fft.fftshift(fft_pattern)), cmap='gray')
    else:
        plt.imsave(f'uniform/pattern_fft/{w}.jpg',
                           np.abs(np.fft.fftshift(fft_pattern)), cmap='gray')


    # Multiply the FFTs elementwise
    product_fft = fft_arr * fft_pattern

    # Take the inverse FFT of the product
    product = np.fft.ifft2(product_fft)

    # Apply fftshift to center the result
    product = np.fft.fftshift(product)


    # Display arr
    # plt.imshow(arr, cmap='gray')
    # plt.show()
    
    if w < 10:
        plt.imsave(f'uniform/arr_img/0{w}.jpg',
                               arr, cmap='gray')
    else:
        plt.imsave(f'uniform/arr_img/{w}.jpg',
                           arr, cmap='gray')
    

    # Display the modulus of the FFT of the array
    # plt.imshow(np.abs(np.fft.fftshift(fft_arr)), cmap='gray')
    # plt.show()
    
    if w < 10:
        plt.imsave(f'uniform/arr_fft/0{w}.jpg',
                               np.abs(np.fft.fftshift(fft_arr)), cmap='gray')
    else:
        plt.imsave(f'uniform/arr_fft/{w}.jpg',
                           np.abs(np.fft.fftshift(fft_arr)), cmap='gray')


    # Display the resulting image
    # plt.imshow(np.abs(product), cmap='gray')
    # plt.show()
    if w < 10:
        plt.imsave(f'uniform/conv_img/0{w}.jpg',
                               np.abs(product), cmap='gray')
    else:
        plt.imsave(f'uniform/conv_img/{w}.jpg',
                           np.abs(product), cmap='gray')


    # Display the modulus of the FFT of the pattern
    # plt.imshow(np.abs(np.fft.fftshift(product_fft)), cmap='gray')
    # plt.show()
    
    if w < 10:
        plt.imsave(f'uniform/conv_fft/0{w}.jpg',
                               np.abs(np.fft.fftshift(product_fft)), cmap='gray')
    else:
        plt.imsave(f'uniform/conv_fft/{w}.jpg',
                           np.abs(np.fft.fftshift(product_fft)), cmap='gray')
    
    
    
    return uniform_arr



for i in range(1,31):
    generate_uniform_interval_ones(i)
    generate_random_ones(i)


    

    
directories = ['random/pattern_img', 'random/pattern_fft', 'random/arr_img', 'random/arr_fft', 
               'random/conv_img', 'random/conv_fft',
               'uniform/pattern_img', 'uniform/pattern_fft', 'uniform/arr_img', 'uniform/arr_fft', 
               'uniform/conv_img', 'uniform/conv_fft']


for directory in directories:

    # using a raw string

    # directory = r"C:\Users\Lyle\Desktop\Computation and Modelling\fourier-transform\forensics\outwards\clipped_img"


    # create a list of image file names in the directory
    image_files = os.listdir(directory)

    # sort the list of image file names to ensure proper ordering in the GIF
    image_files.sort()

    # define the output GIF file name
    output_file =  output_file = directory.replace("/", "_") + ".gif"

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

    
