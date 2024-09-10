import cv2 ; import numpy as np ; import tensorflow as tf ; from tensorflow import keras

file = open("EmotionalFaces.json" , "r")
use = file.read()
file.close()
model = keras.models.model_from_json(use)

model.load_weights("EmotionalFaces.h5")
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1 , 48 , 48 , 1)
    return feature/255.0

webcam = cv2.VideoCapture(0)
Labels = {0:"angry" , 1:"disgusted" , 2:"fearful" , 3:"happy" , 4:"neutral" , 5:"sad" , 6:"suprised"}

while True:
    i , im = webcam.read()
    gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im , 1.3 , 5)

    try:
        if cv2.waitKey(1) & 0XFF == ord("q"):
            break
        for(p,q,r,s) in faces:
            image = gray[q:q+s , p:p+r]
            cv2.rectangle(im , (p , q) , (p+r , q+s) , (255 , 0 , 0) , 2)
            image = cv2.resize(image , (48 , 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = Labels[pred.argmax()]
            
            cv2.putText(im , f"{prediction_label}" , (p-10 , q-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 2 , (255,0,0) , 1)

        cv2.imshow("Output" , im)
        cv2.waitKey(27)
    except cv2.error:
        pass