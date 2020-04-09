import cv2
import tensorflow as tf 
import numpy as np

data_path = '/Users/varun/Documents/Deep_Learning/data/emorecgwoaug.h5'

model = tf.keras.models.load_model(data_path)

haarcascade_path = '/Users/varun/Documents/Deep_Learning/haarcascades/haarcascade_frontalface_default.xml'


face_cascade = cv2.CascadeClassifier(haarcascade_path)

cam = cv2.VideoCapture(0)

class_labels = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

while(True):
    ret,frame = cam.read()
    
    faces = face_cascade.detectMultiScale(frame,1.5)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        roi = tf.cast(roi_gray,tf.float32)/255
        roi = np.expand_dims(roi,axis=0)
        roi = np.expand_dims(roi,axis=3)
        prediction = model.predict(roi)
        result=class_labels[preds.argmax()]
        label_position = (x,y)
        cv2.putText(frame,result,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
    cv2.imshow('webcam',frame)
    if(cv2.waitKey(1)&0XFF == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()