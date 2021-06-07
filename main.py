
import numpy as np  # Numpy is for Array Handler
# Main Module Import From OpenCV Module
import cv2
import sys  # System
from keras.preprocessing import image  
import keras

model=keras.models.load_model("model.h5")

# Path Here Live Detection Face Capture File
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  

# Open Frontal Camera ( Live Web Camera )
video_camera = cv2.VideoCapture(0)

# Live Streaming Face Detection
while True:
    # Face Frame
    ret, frame = video_camera.read()
    # Face Frame For Detected Face
    grayfaces = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Multi Scale For Multi Faces
    faces = faceCascade.detectMultiScale(
        grayfaces,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(40, 40),
        # resize = (600, 600),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # print(faces)
    # print(type(faces))
    # Created Rectangle For Face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 225), 2)
        roi_gray=grayfaces[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(82,82))  
        img_pixels = image.img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  
  
        predictions = model.predict(img_pixels)  
        print(predictions)
        if predictions[0][0]>0.40:
        	max_index=0
        else:
        	max_index=1
        labels=['WithoutMask','WithMask']
        predicted_label=labels[max_index]
        cv2.putText(frame, predicted_label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # Show Image For Live Face Detected
    # windows function
    cv2.imshow('FaceDetection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release function
cv2. video_capture.release()
# Destroyed Open Windows And Kill Memory Storage
cv2.destroyAllWindows()





# while True:  
#     ret,test_img=cap.read()# captures frame and returns boolean value and captured image  
#     if not ret:  
#         continue  
#     gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)  
  
#     faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5) 
#     for (x,y,w,h) in faces_detected:  
#         cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
#         roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
#         cv2.imshow('FaceDetection', test)
#         roi_gray=cv2.resize(roi_gray,(82,82))  
#         img_pixels = image.img_to_array(roi_gray)  
#         img_pixels = np.expand_dims(img_pixels, axis = 0)  
#         img_pixels /= 255  
  
#         predictions = model.predict(img_pixels)  
  
#         #find max indexed array  
#         max_index = np.argmax(predictions[0])

#         print(max_index)