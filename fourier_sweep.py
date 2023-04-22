import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import imageio



def fourier_mask(image_path, inner_radius, outer_radius, sweep_type):
    
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

    
     
    # for rendering
    
    
    
    
    if sweep_type == "growing_outwards":
    

        if outer_radius < 10:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_00{outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_00{outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_00{outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_00{outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_00{outer_radius}.jpg', dpi = 200)


        elif outer_radius < 100:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_0{outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_0{outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_0{outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_0{outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_0{outer_radius}.jpg', dpi = 200)



        else:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_{outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_{outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_{outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_{outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_{outer_radius}.jpg', dpi = 200)


        return img_clipped

    


    if sweep_type == "shrinking_outwards":

    

        if inner_radius < 10:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_00{inner_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_00{inner_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_00{inner_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_00{inner_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_00{inner_radius}.jpg', dpi = 200)


        elif inner_radius < 100:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_0{inner_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_0{inner_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_0{inner_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_0{inner_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_0{inner_radius}.jpg', dpi = 200)



        else:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_{inner_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_{inner_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_{inner_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_{inner_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_{inner_radius}.jpg', dpi = 200)


        return img_clipped

    

    
    
    if sweep_type == "growing_inwards":
    
        dummy_inner_radius = 350 - inner_radius
    
    
        if dummy_inner_radius < 10:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_00{dummy_inner_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_00{dummy_inner_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_00{dummy_inner_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_00{dummy_inner_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_00{dummy_inner_radius}.jpg', dpi = 200)
            
            
            
        elif dummy_inner_radius < 100:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_0{dummy_inner_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_0{dummy_inner_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_0{dummy_inner_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_0{dummy_inner_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_0{dummy_inner_radius}.jpg', dpi = 200)
            
        else:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_{dummy_inner_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_{dummy_inner_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_{dummy_inner_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_{dummy_inner_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_{dummy_inner_radius}.jpg', dpi = 200)
            
    
    
    
    
    
    
    if sweep_type == "shrinking_inwards":
    
   
    
        dummy_outer_radius = 350 - outer_radius
    
    
        if dummy_outer_radius < 10:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_00{dummy_outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_00{dummy_outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_00{dummy_outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_00{dummy_outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_00{dummy_outer_radius}.jpg', dpi = 200)
            
            
            
        elif dummy_outer_radius < 100:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_0{dummy_outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_0{dummy_outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_0{dummy_outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_0{dummy_outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_0{dummy_outer_radius}.jpg', dpi = 200)
            
            
            
        else:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_{dummy_outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_{dummy_outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_{dummy_outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_{dummy_outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_{dummy_outer_radius}.jpg', dpi = 200)
            
    
    
    if sweep_type == "shell_outwards":
    
        if outer_radius < 10:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_00{outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_00{outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_00{outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_00{outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_00{outer_radius}.jpg', dpi = 200)


        elif outer_radius < 100:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_0{outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_0{outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_0{outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_0{outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_0{outer_radius}.jpg', dpi = 200)



        else:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_{outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_{outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_{outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_{outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_{outer_radius}.jpg', dpi = 200)


        return img_clipped

    
    

    if sweep_type == "shell_inwards":

        dummy_outer_radius = 350 - outer_radius
        
        
        if dummy_outer_radius < 10:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_00{dummy_outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_00{dummy_outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_00{dummy_outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_00{dummy_outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_00{dummy_outer_radius}.jpg', dpi = 200)


        elif dummy_outer_radius < 100:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_0{dummy_outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_0{dummy_outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_0{dummy_outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_0{dummy_outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_0{dummy_outer_radius}.jpg', dpi = 200)



        else:
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_img/clipped_image_{dummy_outer_radius}.jpg', img_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_magnitude/clipped_magnitude_spectrum_{dummy_outer_radius}.jpg', 
                       magnitude_spectrum_clipped, cmap='gray')
            plt.imsave(f'{image_path}_{sweep_type}_rendering/clipped_phase/clipped_phase_spectrum_{dummy_outer_radius}.jpg',
                       phase_spectrum_clipped, cmap='gray')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(magnitude_spectrum_clipped.shape[1]), np.arange(magnitude_spectrum_clipped.shape[0]))
            ax.plot_surface(X, Y, magnitude_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_magnitude/3d_clipped_magnitude_spectrum_{dummy_outer_radius}.jpg', dpi = 200)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, phase_spectrum_clipped, cmap='magma')
            fig.savefig(f'{image_path}_{sweep_type}_rendering/clipped_3d_phase/3d_clipped_phase_spectrum_{dummy_outer_radius}.jpg', dpi = 200)


        return img_clipped


        
        
        
        
        
        
        
        

def render(directory):

    # create a list of image file names in the directory
    image_files = os.listdir(directory)

    # sort the list of image file names to ensure proper ordering in the GIF
    image_files.sort()

    # define the output GIF file name
    output_file = f"{directory}.gif"

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
            
                   
                
                  
                

def sweep(image_path, sweep_type):
        
    # Create the main directory
    directory = f'{image_path}_{sweep_type}_rendering'
    os.makedirs(directory)

    # Create the subdirectories
    subdirectory1 = os.path.join(directory, f'clipped_img')
    os.makedirs(subdirectory1)

    subdirectory2 = os.path.join(directory, f'clipped_magnitude')
    os.makedirs(subdirectory2)


    subdirectory3 = os.path.join(directory, f'clipped_phase')
    os.makedirs(subdirectory3)

    subdirectory4 = os.path.join(directory, f'clipped_3d_magnitude')
    os.makedirs(subdirectory4)


    subdirectory5 = os.path.join(directory, f'clipped_3d_phase')
    os.makedirs(subdirectory5)

    

    if sweep_type == "growing_outwards":

        for i in range (0,350,5):
            fourier_mask(image_path,0,i, sweep_type)

    

    if sweep_type == "shrinking_outwards":

        for i in range (0,350,5):
            fourier_mask(image_path,i,350, sweep_type)
    
    
    
    
    if sweep_type == "growing_inwards":
    
        for i in range (0,350,5):
            fourier_mask(image_path,350-i,350, sweep_type)
    
    
    
    if sweep_type == "shrinking_inwards":
    
        for i in range (0,350,5):
            fourier_mask(image_path,0 ,350-i, sweep_type)
    
    
    if sweep_type == "shell_outwards":
        
        for i in range (0,350,5):
            fourier_mask(image_path,0+i ,30 + i, sweep_type)
    
    
    if sweep_type == "shell_inwards":
    
        for i in range (0,350,5):
            fourier_mask(image_path, 350 - i ,380 - i)
    
    
    
    
    render(subdirectory1)
    render(subdirectory2)
    render(subdirectory3)
    render(subdirectory4)
    render(subdirectory5)


if __name__ == "__main__":
    
    # get the command-line arguments
    image_path = sys.argv[1]
    sweep_type = sys.argv[2]
    
    # call the function with the parameters
    sweep(image_path, sweep_type)


