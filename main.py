import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model(r'new.h5')  
img_height = model.input_shape[1]
img_width = model.input_shape[2]


labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p','q','r','s','t','u','v','w','x','y','z'] # Adjust as needed

cap = cv2.VideoCapture(0)

roi_top = 100
roi_bottom = 300
roi_left = 150
roi_right = 350

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]
    if roi.any():
        try:
            resized_roi = cv2.resize(roi, (img_width, img_height))
            img_array = image.img_to_array(resized_roi) / 255.0  
            img_array = np.expand_dims(img_array, axis=0)  
            
            predictions = model.predict(img_array)

            predicted_class_index = np.argmax(predictions[0])
            predicted_label = labels[predicted_class_index]
            confidence = predictions[0][predicted_class_index]

            text = f"Sign: {predicted_label} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error during prediction: {e}")

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()