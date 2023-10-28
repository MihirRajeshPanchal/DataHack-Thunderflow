import cv2
import streamlit as st
import openai
import requests
import mediapipe as mp
from scripts.Squat.squat import ProcessFrame
from scripts.Squat.thresholds import get_thresholds_beginner, get_thresholds_pro
from scripts.Shoulder_Press.rep_counter import rep_counter
from scripts.BicepCurler.bicep import bicep_curl
from scripts.Pushup.PushUpCounter import pushups
from model.GestureDetector.handdetect import handDetect
from dashboard import dashboard_streamlit
from feed import feed_streamlit
from ttsvoice import tts
import pickle
import numpy as np

def dashboard():
    st.subheader("Dashboard")
    dashboard_streamlit()

def push_ups():
    st.subheader("Push Ups")
    pushups()

def shoulder_press():
    st.subheader("Shoulder Press")
    rep_counter()

def calorie_counter():
    st.subheader("Calorie Counter")
    food_calories = {
        "butter_chicken": 350,
        "qubani_ka_meetha": 400,
        "kachori": 250,
        "makki_di_roti_sarson_da_saag": 150,
        "chicken_tikka": 200,
        "paneer_butter_masala": 300,
        "imarti": 450,
        "cham_cham": 350,
        "kofta": 250,
        "poha": 150,
        "modak": 200,
        "pithe": 250,
        "misti_doi": 200,
        "sandesh": 300,
        "adhirasam": 250,
        "aloo_matar": 200,
        "poornalu": 250,
        "kadai_paneer": 300,
        "chicken_tikka_masala": 200,
        "bandar_laddu": 450,
        "chana_masala": 250,
        "gavvalu": 200,
        "unni_appam": 250,
        "lyangcha": 350,
        "chak_hao_kheer": 400,
        "bhatura": 250,
        "kuzhi_paniyaram": 200,
        "aloo_methi": 200,
        "palak_paneer": 300,
        "mysore_pak": 450,
        "misi_roti": 200,
        "karela_bharta": 250,
        "ghevar": 400,
        "sheer_korma": 350,
        "chapati": 150,
        "ledikeni": 250,
        "gajar_ka_halwa": 450,
        "sutar_feni": 400,
        "dal_makhani": 300,
        "shrikhand": 200,
        "navrattan_korma": 350,
        "dum_aloo": 200,
        "daal_baati_churma": 250,
        "dal_tadka": 200,
        "jalebi": 450,
        "maach_jhol": 250,
        "rasgulla": 200,
        "lassi": 150,
        "pootharekulu": 200,
        "bhindi_masala": 250,
        "sohan_papdi": 400,
        "kalakand": 450,
        "aloo_gobi": 200,
        "doodhpak": 200,
        "malapua": 250,
        "ariselu": 200,
        "shankarpali": 350,
        "phirni": 400,
        "litti_chokha": 250,
        "chicken_razala": 200,
        "gulab_jamun": 450,
        "biryani": 350,
        "kajjikaya": 200,
        "sohan_halwa": 400,
        "ras_malai": 350,
        "aloo_tikki": 200,
        "dharwad_pedha": 450,
        "sheera": 200,
        "anarsa": 250,
        "chhena_kheeri": 200,
        "basundi": 250,
        "kakinada_khaja": 400,
        "rabri": 200,
        "naan": 150,
        "kadhi_pakoda": 250,
        "chikki":500,
        "aloo_shimla_mirch":150,
        "double_ka_meetha":400,
        "daal_puri":200,
        "boondi":400,
    }

    with open('data/foodcalorie.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    image_path = 'data/1.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (120, 120))  # Resize to match model's input dimensions
    image = image / 255.0  # Normalize the pixel values
    predictions = model.predict(np.array([image]))
    predicted_class = np.argmax(predictions)
    class_labels = list(food_calories.keys())
    predicted_label = class_labels[predicted_class]
    calorie_value = food_calories.get(predicted_label, "unknown")
    print(f"Your food {predicted_label} has approximately {calorie_value} calories")
    st.write(predicted_label,calorie_value)

def music():
    st.subheader("Music")
    handDetect()

def chat_recommendation():
    st.subheader("Chat Recommendation")

def bicep_curler():
    st.subheader("Bicep Curler")
    bicep_curl()

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

def main():
    st.sidebar.header('Options')
    option = st.sidebar.selectbox('Select an Option', ('Dashboard','Feed','Squat Analyzer','Shoulder Press','Bicep Curler','Push Ups','Calorie Counter','Music','Recommendation'))
    st.sidebar.markdown('Made by Thunderflow for DataHack 2.0')

    if option=="Dashboard":
        dashboard()
    elif option=="Feed":
        res = feed_streamlit()
        option=res
    elif option=="Squat Analyzer":
        on = st.sidebar.toggle('Activate Pro Mode')
        squat_analyzer(on)
    elif option=="Shoulder Press":
        shoulder_press()
    elif option=="Bicep Curler":
        bicep_curler()
    elif option=="Push Ups":
        push_ups()
    elif option=="Calorie Counter":
        calorie_counter()
    elif option=="Music":
        music()
    elif option=="Recommendation":
        chat_recommendation()

st.set_page_config(
    page_title="Thunderflow",
    page_icon="💫",
    layout="wide", 
)
if __name__ == '__main__':
    main()