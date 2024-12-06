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

# data_dir = '/mnt/d/projects/minor_project/data_set/face_recognition_data_images' # path working for wsl2
data_dir = 'D:/projects/minor_project/data_set/face_recognition_data_images' # path working for windows

class_indices = get_class_indices(data_dir)

# Load the model
model = tf.keras.models.load_model("models/facial_recognition_model.keras")

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preprocess a single frame
def preprocess_frame(frame):
    resized = cv2.resize(frame, (224, 224))  # Match model input size
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# Initializing a pandas df for tracking attendance in a csv file
def load_attendance_df(attendance_file):
    if os.path.exists(attendance_file):
        return pd.read_csv(attendance_file)
    return pd.DataFrame(columns=['Name', 'Date', 'Time'])

# Function to run inference
def run_inference():
    attendance_file = 'attendance.csv'
    attendance_df = load_attendance_df(attendance_file)
    
    # Real-time prediction
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    frame_rate = 10  # Limit to 10 frames per second
    frame_interval = 1.0 / frame_rate
    prev_frame_time = datetime.now()

    try:
        while True:
            current_time = datetime.now()
            time_elapsed = (current_time - prev_frame_time).total_seconds()
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
        
            if time_elapsed > frame_interval:
                prev_frame_time = current_time

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (x, y, w, h) in faces:
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Extract and preprocess the face region
                    face = frame[y:y+h, x:x+w]
                    try:
                        processed = preprocess_frame(face)
                        prediction = model.predict(processed)
                        predicted_class_index = np.argmax(prediction)
                        predicted_class_name = class_indices.get(predicted_class_index, "Unknown")
                        confidence = float(prediction[0][predicted_class_index])
                            
                        print(f"Predicted Class: {predicted_class_name}")

                        # Mark attendance for high-confidence predictions
                        if predicted_class_name != "Unknown":# and confidence > 0.7:
                            date_str = current_time.strftime("%Y-%m-%d")
                            time_str = current_time.strftime("%H:%M:%S")

                            # Check if entry already exists for today
                            today_entries = attendance_df[
                                    (attendance_df['Name'] == predicted_class_name) & 
                                    (attendance_df['Date'] == date_str)
                            ]

                            if today_entries.empty:
                                    new_entry = pd.DataFrame([{
                                        'Name': predicted_class_name, 
                                        'Date': date_str, 
                                        'Time': time_str
                                    }])
                                    attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                                    # Save only when new entry is added
                                    attendance_df.to_csv(attendance_file, index=False)
                                    print(f"Marked attendance for {predicted_class_name}")


                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        continue

                    # Display the predicted class name on the frame
                    label = f"{predicted_class_name} "

                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
                # Display the resulting frame
                cv2.imshow("Video Feed", frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit command received")
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

# # Marking the attendance and saving the attendance DataFrame to a CSV file
# attendance_df.to_csv(attendance_file, index=False)
# print("Attendance updated and saved to attendance.csv")

# Run the inference        
# run_inference()