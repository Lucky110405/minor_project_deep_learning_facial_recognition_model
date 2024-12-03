# Capturing the images of students using webcam for facial recognition, predict the attendance of students using the trained CNN model, and mark their attendance.

import cv2
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Creating a Class Index to Class Name Mapping:
def get_class_indices(data_dir):
    class_names = sorted(os.listdir(data_dir))
    class_indices = {i: class_name for i, class_name in enumerate(class_names)}
    return class_indices

data_dir = '/mnt/d/projects/minor_project/face_recognition_data_images'
class_indices = get_class_indices(data_dir)

# Load the model
model = tf.keras.models.load_model("facial_recognition_model.keras")

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocess a single frame
def preprocess_frame(frame):
    resized = cv2.resize(frame, (128, 128))  # Match model input size
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Initializing a pandas df for tracking attendance in a csv file
attendance_file = 'attendance.csv'
if os.path.exists(attendance_file):
    attendance_df = pd.read_csv(attendance_file)
else:
    attendance_df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

# Function to run inference
def run_inference():

    # Real-time prediction
    cap = cv2.VideoCapture(0)
    frame_rate = 10  # Limit to 10 frames per second
    prev = 0

    while True:
        time_elapsed = cv2.getTickCount() - prev
        ret, frame = cap.read()
        if not ret:
            break
        if time_elapsed > (cv2.getTickFrequency() / frame_rate):
            prev = cv2.getTickCount()

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract the face region
                face = frame[y:y+h, x:x+w]
                
                # Preprocess the face region
                processed = preprocess_frame(face)
                print("Frame processed")

                try:
                    prediction = model.predict(processed)
                    predicted_class_index = np.argmax(prediction)
                    predicted_class_name = class_indices.get(predicted_class_index, "Unknown")
                    print(f"Predicted Class: {predicted_class_name}")

                    # Mark attendance
                    if predicted_class_name != "Unknown":
                        now = datetime.now()
                        date_str = now.strftime("%Y-%m-%d")
                        time_str = now.strftime("%H:%M:%S")
                        new_entry = pd.DataFrame([{'Name': predicted_class_name, 'Date': date_str, 'Time': time_str}])
                        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                        attendance_df.drop_duplicates(subset=['Name', 'Date'], keep='last', inplace=True)  # Ensure only the latest entry for each person per day

                except Exception as e:
                    print(f"Prediction error: {e}")

                # Display the predicted class name on the frame
                cv2.putText(frame, f"Predicted: {predicted_class_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        
            # Display the resulting frame
            cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Marking the attendance and saving the attendance DataFrame to a CSV file
attendance_df.to_csv(attendance_file, index=False)
print("Attendance updated and saved to attendance.csv")