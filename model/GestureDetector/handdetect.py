import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import streamlit as st

def handDetect():

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    model = load_model('model/GestureDetector/model')

    f = open('model/GestureDetector/gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()

    image_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        x, y, c = frame.shape

        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        
        className = ''

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

  
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        image_placeholder.image(frame, channels="BGR", use_column_width=True)

        if cv2.waitKey(1) == ord('q'):
            break
        print(className)

    cap.release()

    cv2.destroyAllWindows()
