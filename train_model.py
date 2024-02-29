import os
import cv2
import numpy as np
from PIL import Image

model=cv2.face.LBPHFaceRecognizer_create()
path='./dataset'


def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        try:
             ID=int(os.path.split(imagePath)[-1].split('.')[1])
        except ValueError:
                print('No ID found',imagePath,os.path.split(imagePath)[-1].split('.')[1])
                continue
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("Model training",faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces

ids,faces=getImagesWithID(path)
model.train(faces,ids)
model.save('models/trainModel.yml')
cv2.destroyAllWindows()