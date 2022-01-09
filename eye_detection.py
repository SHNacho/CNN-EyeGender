import numpy as np
import cv2
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy

#cv2.data requiere pip3 install opencv-contrib-python

model = load_model('./eye_gendre.h5')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
ojo = None
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w+10, y+h+10), (0, 255, 0), 2)
        ojo = frame[y:y+h, x:x+w]

        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray, 1.25, 5)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew + 10, ey + eh + 10), (0, 255, 0), 2)
        #     ojo = roi_color[ey:ey+eh, ex:ex+ew]
        
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

while cv2.waitKey(1) != ord('w'):
    cv2.imshow('ojo', ojo[2:-2 , 2:-2])

print(ojo)
ojo = cv2.resize(ojo[2:-2 , 2:-2], (75, 75), interpolation = cv2.INTER_AREA)
ojo_tensor = image.img_to_array(ojo)
ojo_tensor = np.expand_dims(ojo, axis=0)
pred = model.predict(ojo_tensor)
print(pred)
cap.release()
cv2.destroyAllWindows
