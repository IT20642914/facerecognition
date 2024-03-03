import os
import cv2
import numpy as np
from PIL import Image

model = cv2.face.LBPHFaceRecognizer_create()
path = './dataset'
MIN_FACE_DIMENSIONS = (1000, 900)  # Set your minimum desired dimensions


def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    names = {}  # Dictionary to store user names with IDs
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
         # Dimension check
        height, width = faceNp.shape
        if height < MIN_FACE_DIMENSIONS[0] or width < MIN_FACE_DIMENSIONS[1]:
            print(f"Image {imagePath} is too small. Skipping...")
            continue  # Skip to the next image

        try:
            user_id = int(os.path.split(imagePath)[-1].split('.')[0])
            name = os.path.split(imagePath)[-1].split('.')[2]
        except ValueError:
            print('No ID found', imagePath, os.path.split(imagePath)[-1].split('.')[1])
            continue
        faces.append(faceNp)
        ids.append(user_id)
        names[user_id] = name  # Map ID to name
        cv2.imshow("Model training", faceNp)
        cv2.waitKey(10)
    return ids, faces, names

ids, faces, names = getImagesWithID(path)
model.train(faces, np.array(ids))

# Save the dictionary of names to a file
with open('models/names_mapping.txt', 'w') as file:
    for id, name in names.items():
        file.write(f"{id}:{name}\n")

model.save('models/trainModel.yml')
cv2.destroyAllWindows()
