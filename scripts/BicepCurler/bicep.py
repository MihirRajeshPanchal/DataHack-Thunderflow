import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from ttsvoice import tts
def bicep_curl():
    def calc_angle(a,b,c): 
        ''' Arguments:
            a,b,c -- Values (x,y,z, visibility) of the three points a, b and c which will be used to calculate the
                    vectors ab and bc where 'b' will be 'elbow', 'a' will be shoulder and 'c' will be wrist.
            
            Returns:
            theta : Angle in degress between the lines joined by coordinates (a,b) and (b,c)
        '''
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        ab = np.subtract(a, b)
        bc = np.subtract(b, c)
        
        theta = np.arccos(np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(bc)))     
        theta = 180 - 180 * theta / 3.14    
        return np.round(theta, 2)


    def infer():
        mp_drawing = mp.solutions.drawing_utils     
        mp_pose = mp.solutions.pose                 
        left_flag = None     
        left_count = 0       
        right_flag = None
        right_count = 0

        image_placeholder = st.empty()
        cap = cv2.VideoCapture(0)
        pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) 
        while cap.isOpened():
            _, frame = cap.read()

            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      
            image.flags.writeable = False
            
            
            results = pose.process(image)                       

            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      

            try:
                
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                
                left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)      
                right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

                
                cv2.putText(image,\
                        str(left_angle), \
                            tuple(np.multiply([left_elbow.x, left_elbow.y], [640,480]).astype(int)),\
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
                cv2.putText(image,\
                        str(right_angle), \
                            tuple(np.multiply([right_elbow.x, right_elbow.y], [640,480]).astype(int)),\
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
            
                
                if left_angle > 160:
                    left_flag = 'down'
                if left_angle < 50 and left_flag=='down':
                    left_count += 1
                    left_flag = 'up'
                    left_res="Left Repppppppppppp"+str(left_count)
                    tts(left_res)

                if right_angle > 160:
                    right_flag = 'down'
                if right_angle < 50 and right_flag=='down':
                    right_count += 1
                    right_flag = 'up'
                    right_res="Right Repppppppppppp"+str(right_count)
                    tts(right_res)
                
            except:
                pass

            
            cv2.rectangle(image, (0,0), (1024,73), (128, 128, 128), -1)
            cv2.putText(image, 'Left:' + str(left_count) + '    Right:' + str(right_count),
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            image_placeholder.image(image, channels="BGR", use_column_width=True)
            # cv2.imshow('MediaPipe feed', image)

            k = cv2.waitKey(30) & 0xff  
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            elif k==ord('r'):
                left_count = 0
                right_count = 0

        cap.release()
        cv2.destroyAllWindows()
    infer()