import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Load the image
image_path = "C:\\Users\\HP\\Desktop\\Smile\\face.jpg"
org = cv2.imread(image_path)
img = cv2.imread(image_path)

# Convert the original image to have an alpha channel
org = cv2.cvtColor(org, cv2.COLOR_BGR2BGRA)

# Initialize the FaceMesh detector
detector = FaceMeshDetector(maxFaces=1)

# Detect the face and landmarks
im, faces = detector.findFaceMesh(img, draw=True)

# Define the indices for the mouth region (based on the FaceMesh model)
mouth_indices = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
    181, 91, 146
]

# Check if a face was detected
if faces:
    face = faces[0]
    
    # Get the mouth landmarks
    mouth_landmarks = [face[idx] for idx in mouth_indices]
    
    # Convert landmarks to a numpy array
    mouth_landmarks_np = np.array(mouth_landmarks, np.int32)
    mouth_landmarks_np = mouth_landmarks_np.reshape((-1, 1, 2))
    
    # Create a mask with the same dimensions as the original image
    mask = np.zeros_like(org[:, :, 0], dtype=np.uint8)
    
    # Fill the polygon defined by the mouth landmarks in the mask
    cv2.fillPoly(mask, [mouth_landmarks_np], 255)
    
    # Invert the mask to get the area outside the mouth region
    mask_inv = cv2.bitwise_not(mask)
    
    # Apply the mask to the alpha channel of the original image
    org[:, :, 3] = cv2.bitwise_and(org[:, :, 3], mask_inv)

# Save the result
cv2.imwrite("C:\\Users\\HP\\Desktop\\Smile\\face_with_transparent_mouth.png", org)

# Display the image with the transparent mouth region
cv2.imshow("Image with Transparent Mouth Region", org)
cv2.waitKey(0)
cv2.destroyAllWindows()
