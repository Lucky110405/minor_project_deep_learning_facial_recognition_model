import streamlit as st
import pandas as pd
from inference import run_inference

# Title of the web app
st.title('Deep Learning Model Deployment with Webcam')

# Button to start webcam
if st.button('Start Webcam'):
    run_inference()

# Button to refresh attendance data
if st.button('Refresh Attendance Data'):
    df = pd.read_csv('attendance.csv')
    st.write(df)