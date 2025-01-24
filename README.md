# minor_project_deep_learning_facial_recognition_model

Creating a deep learning facial recognition model for automating classroom attendance.

Our primary objective with this project was to streamline teacher responsibilities by automating attendance tracking by developing a real-time facial recognition system using CNN.	

This is my first full project that i had made and so there may still be many shortcomings to this project, but i had tried my best in building this project. 

I am always eager and ready to learn. I welcome everyone to help improve this project by suggesting what more should be done or pointing out any mistakes. Any tips for improvement are greatly appreciated.

# Automated Attendance System Working

This project is a deep learning-based facial recognition model designed to automate classroom attendance. The system uses a webcam to capture images, processes them using a pre-trained model, and marks attendance by recognizing faces.

Initially, a custom Convolutional Neural Network (CNN) model was designed and trained for facial recognition. However, the results were not satisfactory. To improve the performance, transfer learning was used using the EfficientNetB0 model pre-trained on the ImageNet dataset. Both the custom CNN model and the transfer learning model training files have been pushed to the remote repository for further discussion and enhancements.

## Features

- **Webcam Integration**: Captures images using a webcam.
- **Facial Recognition**: Uses a deep learning model to recognize faces.
- **Automated Attendance**: Marks attendance automatically based on recognized faces.
- **Streamlit Interface**: Provides a user-friendly web interface for interaction.

## Project Structure

- **app.py**: Main Streamlit application file.
- **inference.py**: Contains the inference logic for facial recognition.
- **train_CNN_model.py**: Script for training the custom CNN model.
- **training_CNN_by_transfer_learning.py**: Script for training the model using transfer learning.
- **data_set**: Directory containing the dataset for training and testing.
- **attendance.csv**: CSV file where attendance records are stored.

## Usage of Github Copilot

I used GitHub copilot for learning machine learning and making this project. There were many things that I didn't know about, so I utilized gitHub copilot to help me understand those things as well as write code. The code is not fully written through AI, but some parts like inference_app have a good chunk written with the help of github copilot. However, I mostly tried understood everything as to why it was used.

## Future scope

The project has immense potential for expansion beyond automated attendance. Here are the envisioned features that I wish to integrate in near future:

Student Behavior Analysis
Personalized Learning Recommendations
Scalability and Multi-Classroom Management

## Willingness to Learn

Although I might not currently know many things, I am determined to learn and grow. I am willing to put in the effort to understand new concepts and improve my skills. Your feedback and suggestions are invaluable to me on this learning journey.