import cv2
import sys

# Load the image


def get_coordinates(image_path):
    image = cv2.imread(image_path)

    # Create a window to display the image
    cv2.namedWindow('image')

    # Define a callback function to get the coordinates of the cursor
    def get_coords(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print(f"Clicked at: ({x}, {y})")
            print(f"[{x},{y}]")

    # Set the mouse callback function for the window
    cv2.setMouseCallback('image', get_coords)

    # Display the image and wait for a key press
    cv2.imshow('image', image)
    cv2.waitKey(0)

    # Destroy the window and close the program
    cv2.destroyAllWindows()
    
    print("Thank you!")
    
    
if __name__ == "__main__":
    
    # get the command-line arguments
    image_path = sys.argv[1]

    
    # call the function with the parameters
    print("Welcome to FourierViz by Schwarzschlyle")
    print("Please wait while I explore the Fourier information of your image. Thank you!")
    print("Explore your image...")
    
    
    get_coordinates(image_path)
