import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

# Load the image
image_path = "C:\\Users\\HP\\Desktop\\Smile\\face.jpg"
org = cv2.imread(image_path)
img = cv2.imread(image_path)

# Initialize the FaceMesh detector
detector = FaceMeshDetector(maxFaces=1)

# Detect the face and landmarks
im, faces = detector.findFaceMesh(img, draw=True)

# Define the indices for the mouth region (based on the FaceMesh model)
mouth_indices = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
    181, 91, 146, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 
    314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 
    321, 405, 314, 17, 84, 181, 91, 146
]

# # Check if a face was detected
if faces:
    face = faces[0]
    
    # Draw only the mouth landmarks
    for idx in mouth_indices:
        x, y = face[idx]
        cv2.circle(org, (x, y), 2, (0, 255, 0), -1)

# Display the image with mouth landmarks
cv2.imshow("Image with Mouth Landmarks", org)
cv2.waitKey(0)
cv2.destroyAllWindows()



