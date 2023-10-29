import streamlit as st
import time
import cv2
import mediapipe as mp


def lateral_raise():

    

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    
    # cap = cv2.VideoCapture("scripts\ExerciseDetection\Videos\lateral raise.mp4")
    cap = cv2.VideoCapture(0)
        
        
       
    image_placeholder = st.empty()
       
        
       
     
        
        
       
    
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
           
    
        
    
    
        image_placeholder.image(img, channels="BGR", use_column_width=True)
        cv2.waitKey(1)
    