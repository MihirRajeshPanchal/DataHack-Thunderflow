import cv2
import mediapipe as mp
import numpy as np
import posemodule as pm
import streamlit as st
import os
from ttsvoice import tts

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def create_video(output_path, frame_folder):
    
    image_files = os.listdir(frame_folder)
    image_files.sort()

    if len(image_files) == 0:
        print("No image frames found in the specified folder.")
        return


    first_frame = cv2.imread(os.path.join(frame_folder, image_files[0]))
    height, width, layers = first_frame.shape


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for image_file in image_files:
        image_path = os.path.join(frame_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


def shoulder():

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(r"C:/Users/TIRATH/Documents/ThunderFlow/DataHack-Thunderflow/shoulder_sid.mp4")
        frame_folder = r"C:/Users/TIRATH/Documents/ThunderFlow/DataHack-Thunderflow/scripts/Shoulder_Press/frames"  # Create a folder to store frames
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)
        frame_count=0
        image_placeholder = st.empty()
        counter = 0
        stage = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
    
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    res="Rep"+str(counter)
                    tts(res)

            except:
                pass


            cv2.rectangle(image, (0,0), (1024,73), (128, 128, 128), -1)
            # Center-top position for "REPS" and counter
            text = 'REPS'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            cv2.putText(image, text, (text_x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            text = str(counter)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            cv2.putText(image, text, (text_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Center-bottom position for "STAGE"
            text = 'STAGE'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = image.shape[0] - 12
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Center-bottom position for stage
            text_size = cv2.getTextSize(stage, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = image.shape[0] - 60
            cv2.putText(image, stage, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            image_placeholder.image(image, channels="BGR", use_column_width=True)

            frame_count += 1
            frame_filename = f"frame_{frame_count:04d}.png"
            frame_path = os.path.join(frame_folder, frame_filename)
            cv2.imwrite(frame_path, image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# shoulder()
create_video(r"C:/Users/TIRATH/Documents/ThunderFlow/DataHack-Thunderflow/outputs/shoulderpressout2.mp4","frames")