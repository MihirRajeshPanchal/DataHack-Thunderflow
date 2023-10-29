import cv2
import streamlit as st
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
from leaderboard import leaderboard_streamlit
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

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # from tensorflow import keras
        # import numpy as np
        # from PIL import Image
        # model = keras.models.load_model('data/FV.h5')

        # image_path = uploaded_file
        # img = Image.open(image_path)
        # img = img.resize((120, 120))  
        # img = np.array(img)
        # img = img / 255.0  

        # img = np.expand_dims(img, axis=0)

        # predictions = model.predict(img)

        # predicted_class = np.argmax(predictions)

        # class_labels = list(food_calories.keys()) 
        # predicted_label = class_labels[predicted_class]
        from PIL import Image
        from keras.preprocessing.image import load_img, img_to_array
        from keras.models import load_model
        import requests
        from bs4 import BeautifulSoup

        model = load_model('data/FV.h5')
        labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
                7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
                14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
                19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
                26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
                32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

        fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
                'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
        vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
                    'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
                    'Tomato', 'Turnip']


        def fetch_calories(prediction):
            try:
                url = 'https://www.google.com/search?&q=calories in ' + prediction
                req = requests.get(url).text
                scrap = BeautifulSoup(req, 'html.parser')
                calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
                return calories
            except Exception as e:
                st.error("Can't able to fetch the Calories")
                print(e)


        def processed_img(img_path):
            img = load_img(img_path, target_size=(224, 224, 3))
            img = img_to_array(img)
            img = img / 255
            img = np.expand_dims(img, [0])
            answer = model.predict(img)
            y_class = answer.argmax(axis=-1)
            print(y_class)
            y = " ".join(str(x) for x in y_class)
            y = int(y)
            res = labels[y]
            print(res)
            return res.capitalize()
        
        img = Image.open(uploaded_file).resize((250, 250))
        st.image(img, use_column_width=False)
        result = processed_img(uploaded_file)
        if result in vegetables:
            st.info('**Category : Vegetables**')
        else:
            st.info('**Category : Fruit**')
        st.success("**Predicted : " + result + '**')
        cal = fetch_calories(result)
        if cal:
            st.warning('**' + cal + '(100 grams)**')

def leaderboard():
    st.subheader("Leaderboard")
    leaderboard_streamlit()

def music():
    st.subheader("Music")
    handDetect()

def chat_recommendation():
    st.subheader("Chat Recommendation")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        
        st.chat_message("user").markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        
        prompt = prompt.title()

        response = f"Echo: {prompt}"
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})


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
    option = st.sidebar.selectbox('Select an Option', ('Dashboard','Feed','Squat Analyzer','Shoulder Press','Bicep Curler','Push Ups','Calorie Counter','Leaderboard','Recommendation','Music'))
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
    elif option=="Leaderboard":
        leaderboard()
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