import cv2
import mediapipe as mp
import numpy as np
import scripts.Pushup.posemodule as pm
import streamlit as st
import os
from ttsvoice import tts


def pushups():
    image_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    frame_folder = "frames"  # Create a folder to store frames
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
    frame_count=0
    detector = pm.poseDetector()
    count = 0
    direction = 0
    form = 0
    feedback = "Fix Form"

    while cap.isOpened():
        ret, img = cap.read() 
        
        width  = cap.get(3)  
        height = cap.get(4)  
        
        
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) != 0:
            elbow = detector.findAngle(img, 11, 13, 15)
            shoulder = detector.findAngle(img, 13, 11, 23)
            hip = detector.findAngle(img, 11, 23,25)
            
            
            per = np.interp(elbow, (90, 160), (0, 100))
            
            
            bar = np.interp(elbow, (90, 160), (380, 50))

            
            if elbow > 160 and shoulder > 40 and hip > 160:
                form = 1
        
            
            if form == 1:
                if per == 0:
                    if elbow <= 90 and hip > 160:
                        feedback = "Up"
                        if direction == 0:
                            count += 0.5
                            direction = 1
                    else:
                        feedback = "Fix Form"
                        tts(feedback)
                if per == 100:
                    if elbow > 160 and shoulder > 40 and hip > 160:
                        feedback = "Down"
                        if direction == 1:
                            count += 0.5
                            direction = 0
                    else:
                        feedback = "Fix Form"
                        tts(feedback)
                    
                        
        
            print(count)
            
            
            if form == 1:
                cv2.rectangle(img, (580, 50), (600, 380), (128, 128, 128), 3)
                cv2.rectangle(img, (580, int(bar)), (600, 380), (128, 128, 128), cv2.FILLED)
                cv2.putText(img, f'{int(per)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 255, 255), 2)


            
            cv2.rectangle(img, (0, 380), (100, 480), (128, 128, 128), cv2.FILLED)
            cv2.putText(img, str(int(count)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5,
                        (255, 255, 255), 5)
            
            
            cv2.rectangle(img, (500, 0), (640, 40), (128, 128, 128), cv2.FILLED)
            cv2.putText(img, feedback, (500, 40 ), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 255, 255), 2)

            
        image_placeholder.image(img, channels="BGR", use_column_width=True)

        frame_count += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

