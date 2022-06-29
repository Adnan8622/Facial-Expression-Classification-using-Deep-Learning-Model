import cv2
import numpy as np
from keras.models import load_model

faceCascade = cv2.CascadeClassifier('F:/My Library/Fer2013 with code/haarcascade_frontalface_alt2.xml')
model = load_model('F:/My Library/Fer2013 with code/Facial_model.h5')#model load

target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread('F:/My Library/Fer2013 with code/mix faces.jpg')
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(im,scaleFactor=1.1)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 96, 0), 2,5)
    face_crop = im[y:y+h,x:x+w]
    face_crop = cv2.resize(face_crop,(48,48))
    #face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_crop = face_crop.astype('float32')/255
    face_crop = np.asarray(face_crop)
    face_crop = face_crop.reshape(1,face_crop.shape[0],face_crop.shape[1],1)
    res=model.predict(face_crop)
    result = target[np.argmax(model.predict(face_crop))]
    cv2.putText(img,result,(x,y), font, 1, (0,0,200), 3, cv2.LINE_AA)
cv2.imshow('result', img)
#cv2.imwrite('result.jpg',img)
#cv2.imwrite('result.jpg',im)
cv2.waitKey(0)
