import streamlit as st
import pandas as pd
from inference import run_inference
import os

# Title of the web app
st.title('Deep Learning Model Deployment with Webcam')

# Button to start webcam
if st.button('Start Webcam'):
    st.write("Starting webcam and running inference...")
    run_inference()
    st.success("Inference completed successfully.")

# Button to refresh attendance data
if st.button('Refresh Attendance Data'):
    attendance_file = 'attendance.csv'
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        st.write("Attendance Data:")
        st.dataframe(df)
    else:
        st.warning("No attendance data available.")