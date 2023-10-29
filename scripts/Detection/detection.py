from transformers import pipeline
from PIL import Image
import streamlit as st
import time
import cv2
import mediapipe as mp
import numpy as np
from scripts.Shoulder_Press.rep_counter import rep_counter
from scripts.BicepCurler.bicep import bicep_curl
from scripts.Squat.squat import ProcessFrame
from scripts.Squat.thresholds import get_thresholds_beginner, get_thresholds_pro
from scripts.Pushup.PushUpCounter import pushups
from scripts.LateralRaise.lateral_raise import lateral_raise

pipe = pipeline("image-classification", model="siddhantuniyal/rare-puppers")

def squat_analyzer(on):
    st.subheader('Squat Posture Analysis')
    cap = cv2.VideoCapture(0)
    image_placeholder = st.empty()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    if on:
        threshold=get_thresholds_beginner()
    else:
        threshold=get_thresholds_pro()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processor = ProcessFrame(thresholds=threshold, flip_frame=True)
        processed_frame, play_sound = processor.process(frame, pose) 
        image_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

        
    cap.release()

def take_pic():
    time.sleep(3)

def detection():


    picture = st.camera_input("Take a picture" , on_change = take_pic)
    
    if picture:

        img = Image.open(picture)

        results = pipe(img)

        exercise = results[0]["label"]
    
        if exercise=="push up exercise":
            st.subheader("Push Ups")
            pushups()
            
        elif exercise=="shoulder press exercise":
            st.subheader("Shoulder Press")
            rep_counter()

        elif exercise=="bicep curl exercise":
            st.subheader("Bicep Curler")
            bicep_curl()

        elif exercise=="lateral raise exercise":
            st.subheader("Lateral Raise")
            lateral_raise()

        else:

            st.subheader('Squat Posture Analysis')
            on = st.sidebar.toggle('Activate Pro Mode')
            squat_analyzer(on)
            

    



    
    