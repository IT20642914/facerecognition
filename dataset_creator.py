import cv2
import numpy as np
import os

# Face detection setup
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to apply data augmentations
def apply_augmentations(image):
    # Flipping
    image = cv2.flip(image, 1) 

    # Brightness and contrast variations
    alpha = np.random.uniform(0.8, 1.2) 
    beta = np.random.randint(-30, 30) 
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)  

    # Rotations 
    angle = np.random.randint(-10, 10) 
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows))  

    return image

# Get user information
name = input('Enter user name: ')
id = input('Enter user id: ')

# Create dataset subfolder (optional)
user_folder = os.path.join('dataset', f'User.{id}.{name}')
os.makedirs(user_folder, exist_ok=True)  # Create if it doesn't exist

# Start video capture
camera = cv2.VideoCapture(0)
sampleFaces = 0

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if any faces were detected
    if len(faces) > 0: 
        for (x, y, w, h) in faces:
            sampleFaces += 1

            # Extract face image
            face_image = gray[y:y+h, x:x+w]

            # Apply data augmentation:
            augmented_image = apply_augmentations(face_image.copy())

            # Save the augmented image
            save_path = os.path.join(user_folder, f'{sampleFaces}.augmented.jpg')
            cv2.imwrite(save_path, augmented_image)

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.waitKey(100)

    cv2.imshow("Face", img)

    if sampleFaces == 200:
        print("Data set with 200 faces has been successful created")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
