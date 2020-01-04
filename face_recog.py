import numpy as np
import time
import cv2 as cv
#chamo o classificador de rosto
face_classifier = cv.CascadeClassifier('haarcascade-frontalface-default.xml')
# Captura o video principal e transforma em imagem
cap = cv.VideoCapture(0)
# Para rodar sempre
while True:
    # Define a imagem lendo a matriz da camera
    ret, img = cap.read()
    # Coloca em escala em cinza
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Classifica com o dataset
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
   

    cv.imshow('img', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
