import cv2
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# recognizer = cv2.face.LBPHFaceRecognizer_create()

# recognizer.read('model.yaml')
model = load_model('model.h5')
cap = cv2.VideoCapture(0)

# labels = joblib.load('labels.joblib')
# labels = {v:k for k,v in labels.items()}
while True:
    check, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h,x:x+w]
        cv2.imwrite('face.jpg',roi)
        test_image = image.load_img('face.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if np.argmax(result, axis=1) == 0:
            cv2.putText(frame, 'bill gates', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, cv2.LINE_AA)
        elif np.argmax(result, axis=1) == 1:
            cv2.putText(frame, 'ghaff', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, cv2.LINE_AA)   
        else:
            cv2.putText(frame, 'mark', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, cv2.LINE_AA)
            # print(labels[id])

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break