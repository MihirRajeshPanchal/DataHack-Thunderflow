import cv2
import streamlit as st
import openai
import requests
import mediapipe as mp
from scripts.Squat.squat import ProcessFrame
from scripts.Squat.thresholds import get_thresholds_beginner, get_thresholds_pro
from scripts.Shoulder_Press.rep_counter import rep_counter
from model.GestureDetector.handdetect import handDetect

def dashboard():
    st.subheader("Dashboard")

def shoulder_press():
    st.subheader("Shoulder Press")
    rep_counter()

def calorie_counter():
    st.subheader("Calorie Counter")

def music():
    st.subheader("Music")
    handDetect()

def chat_recommendation():
    st.subheader("Chat Recommendation")

def squat_analyzer():
    st.subheader('Squat Posture Analysis')
    cap = cv2.VideoCapture(0)
    image_placeholder = st.empty()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processor = ProcessFrame(thresholds=get_thresholds_beginner(), flip_frame=True)
        processed_frame, play_sound = processor.process(frame, pose)
        image_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

        
    cap.release()

def main():
    st.sidebar.header('Options')
    option = st.sidebar.selectbox('Select an Option', ('Dashboard','Squat Analyzer','Shoulder Press','Calorie Counter','Music','Recommendation'))
    st.sidebar.markdown('Made by Thunderflow for DataHack 2.0')

    if option=="Dashboard":
        dashboard()
    elif option=="Squat Analyzer":
        squat_analyzer()
    elif option=="Shoulder Press":
        shoulder_press()
    elif option=="Calorie Counter":
        calorie_counter()
    elif option=="Music":
        music()
    elif option=="Recommendation":
        chat_recommendation()

if __name__ == '__main__':
    main()