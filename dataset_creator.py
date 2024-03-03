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

# Get video file path and user information
video_path = '20240302_234719.mp4' 
name = input('Enter user name: ')
id = input('Enter user id: ')


# Open the video file
video_capture = cv2.VideoCapture(video_path)

sample_faces = 0
frame_skip_interval = 5  

while sample_faces < 200: 
    ret, frame = video_capture.read()

    if not ret: 
        print("End of video reached and still not enough images. Consider a different video.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            sample_faces += 1
            face_image = gray[y:y+h, x:x+w]
            augmented_image = apply_augmentations(face_image.copy())
            
             # Save original face image
            face_save_path = os.path.join('dataset', f'{id}.{sample_faces}.{name}_face.jpg')
            cv2.imwrite(face_save_path, face_image)
            
            # Save augmented image
            augmented_save_path = os.path.join('dataset', f'{id}.{sample_faces}.{name}_augmented.jpg')
            cv2.imwrite(augmented_save_path, augmented_image)
            
            # Optional Visualization 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
            cv2.imshow("Face Extraction", frame[y:y+h*2, x:x+w*2]) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit option during visualization

print("Dataset with 200 faces has been successfully created") 
video_capture.release()
cv2.destroyAllWindows()
