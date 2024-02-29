import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('models/trainModel.yml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font=cv2.FONT_HERSHEY_SIMPLEX

camera = cv2.VideoCapture(0)    
while True:
    ret, img =camera.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
        if(conf<50):
            if(Id==1):
                Id="avishka"
                cv2.putText(img,str(Id), (x,y+h),font, 1, (0,255,0), 2)
            elif(Id==2):
                Id="sanju"
                cv2.putText(img,str(Id), (x,y+h),font, 1, (0,255,0), 2)
        else:
            Id="Unknown"
        cv2.putText(img,str(Id), (x,y+h),font, 1, (0,255,0), 2)
        
    cv2.imshow('Face Recognizer',img)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()