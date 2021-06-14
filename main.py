
import numpy as np  
import cv2
import sys  
from keras.preprocessing import image  
import keras

model=keras.models.load_model("Model/model.h5")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  

video_camera = cv2.VideoCapture(0)

while True:
    ret, frame = video_camera.read()
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = faceCascade.detectMultiScale(
        face,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(40, 40),
        # resize = (600, 600),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
        roi=face[y:y+w,x:x+h]
        roi=cv2.resize(roi_gray,(82,82))  
        img_pixels = image.img_to_array(roi)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  
  
        predictions = model.predict(img_pixels)  
        print(predictions)
        if predictions[0][0]>0.50:
            max_index=0
        else:
            max_index=1
        labels=['WithoutMask','WithMask']
        predicted_label=labels[max_index]
        cv2.putText(frame, predicted_label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('FaceDetection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release function
cv2. video_capture.release()
cv2.destroyAllWindows()
