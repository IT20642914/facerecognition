import cv2
import numpy as np

# Load the mapping of user IDs to names
names_mapping = {}
with open('models/names_mapping.txt', 'r') as file:
    for line in file:
        user_id, name = line.strip().split(':')
        names_mapping[int(user_id)] = name

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('models/trainModel.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

camera = cv2.VideoCapture(0)
while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is below threshold
        if conf > 50:
            if Id in names_mapping:  # Check if ID exists in the mapping
                name = names_mapping[Id]
            else:
                name = "Unknown"
        cv2.putText(img, str(name), (x, y+h), font, 1, (0, 255, 0), 2)
    cv2.imshow('Face Recognizer', img)
    
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
