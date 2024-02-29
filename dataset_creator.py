import cv2
import numpy as np
import time

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

name = input('Enter user name: ')
id = input('Enter user id: ')
camera = cv2.VideoCapture(0)
sampleFaces=0

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h)in faces:
        sampleFaces+=1
        cv2.imwrite("dataset/User."+str(id)+"."+str(name)+"."+str(sampleFaces)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    if(sampleFaces == 200):
        print("Data set with 200 faces has been successful created")
        break
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
        
camera.release()
cv2.destroyAllWindows()
