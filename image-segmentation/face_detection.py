import cv2
import dlib
import numpy as np

# Load the pre-trained facial landmark detector
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Create a face detector object
detector = dlib.get_frontal_face_detector()



# Define the video capture object
cap = cv2.VideoCapture(0)

# Define the segmentation function
def face_segmentation(frame):
    if frame is None:
        return None

    # Convert the frame from BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the face detector object
    faces = detector(gray, 0)

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Get the largest
    largest_face = max(faces, key=lambda x: x.area())

    # Extract the region of the image
    x, y, w, h = largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()
    face_image = frame[y:y+h, x:x+w]

    return face_image


#     return (coordinates, area, perimeter)
def extract_features(segmented_frame):
    if segmented_frame is None:
        return None

    # Convert the segmented frame to grayscale
    gray = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2GRAY)

    # Detect the facial landmarks using the predictor object
    faces = detector(gray, 0)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])

    # Extract the coordinates of the facial landmarks
    coordinates = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        coordinates[i] = (landmarks.part(i).x, landmarks.part(i).y)

    # Calculate the area and perimeter of the face
    area = cv2.contourArea(coordinates)
    perimeter = cv2.arcLength(coordinates, True)

    return (coordinates, area, perimeter)


# Start the loop to capture and process each frame
while True:
    # Capture a frame
    ret, frame = cap.read()

    # Apply face segmentation
    segmented_frame = face_segmentation(frame)

    # Extract features
    features = extract_features(segmented_frame)

    # Display the results
    cv2.imshow('Original Frame', frame)
    if segmented_frame is not None:
        cv2.imshow('Segmented Frame', segmented_frame)
    if features is not None:
        # Display the coordinates of the facial landmarks
        for (x, y) in features[0]:
            cv2.circle(segmented_frame, (x, y), 1, (0, 255, 0), -1)
        # Display the area and perimeter of the face
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(segmented_frame, 'Area: {:.2f} px'.format(features[1]), (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(segmented_frame, 'Perimeter: {:.2f} px'.format(features[2]), (10, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Features', segmented_frame)
        
        
        
        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break