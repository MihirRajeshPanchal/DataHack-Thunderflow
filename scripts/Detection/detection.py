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

pipe = pipeline("image-classification", model="siddhantuniyal/exercise-detection")

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
def lateralRaise():

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    
    # cap = cv2.VideoCapture("scripts\ExerciseDetection\Videos\lateral raise.mp4")
    cap = cv2.VideoCapture(0)
    
    
   
    
   
    
   
 
    
    
   
    
    up = False
    counter = 0
    
    while True:
        success , img = cap.read()
    
        img = cv2.resize(img , (1280,720))
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            # cv2.putText(img , ltr , (0,50),cv2.FONT_HERSHEY_PLAIN,3,red,12)
            
    
            mp_drawing.draw_landmarks(img , results.pose_landmarks , mp_pose.POSE_CONNECTIONS)
            points = {}
            for id,lm in enumerate(results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                points[id] = (cx,cy)
    
            cv2.circle(img , points[12] , 15 , (255,0,0),cv2.FILLED)
            cv2.circle(img , points[14] , 15 , (255,0,0),cv2.FILLED)
            cv2.circle(img , points[11] , 15 , (255,0,0),cv2.FILLED)
            cv2.circle(img , points[13] , 15 , (255,0,0),cv2.FILLED)
    
    
            if not up and points[14][1] < points[12][1]:
                up = True
                counter+=1
            elif points[14][1] > points[12][1]:
                up = False
    
    
    
    
        cv2.putText(img , str(counter) , (100,150),cv2.FONT_HERSHEY_PLAIN , 12 , (255,0,0),12)
           
    
        
    
    
        cv2.imshow("img",img)
        cv2.waitKey(1)
    

def detection():

    if st.button('Take a picture'):
        
        st.write('Wait for 5 seconds...')
        time.sleep(5) 

    picture = st.camera_input("Take a picture")
    
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
            lateralRaise()

        else:

            st.subheader('Squat Posture Analysis')
            on = st.sidebar.toggle('Activate Pro Mode')
            squat_analyzer(on)
            

    



    
    