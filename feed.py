import streamlit as st
import pandas as pd
# from app import squat_analyzer,shoulder_press,bicep_curler,push_ups
from scripts.Pushup.PushUpCounter import pushups
def feed_streamlit():
    carousel_data = [
        {"image_url": "1.jpg", "name": "Item 1"},
        {"image_url": "1.jpg", "name": "Item 2"},
        {"image_url": "1.jpg", "name": "Item 3"},
    ]
    # col1, col2 = st.columns(2)
    # col3, col4 = st.columns(2)
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.text("Pushups")
        st.image(r"data/Films & TV 29-10-2023 10_32_03.png")
        st.text("Reps : 20")
        if st.button("Like",key=1):
            st.toast('Picture Liked', icon='💓')
        if st.button("Challenge",key=5):
            pushups()

            
    with col2:
        st.text("Squats")
        st.image(r"data/Films & TV 29-10-2023 10_30_25.png")
        st.text("Reps : 30")
        if st.button("Like",key=2):
            st.toast('Picture Liked', icon='💓')
        if st.button("Challenge",key=6):
            pushups()

    with col3:
        st.text("Shoulder Press")
        st.image(r"data/Films & TV 29-10-2023 10_31_20.png")
        st.text("Reps : 15")
        if st.button("Like",key=3):
            st.toast('Picture Liked', icon='💓')
        if st.button("Challenge",key=7):
            pushups()
    
    with col4:
        st.text("Bicep Curler")
        st.image(r"data/Films & TV 29-10-2023 10_32_30.png")
        st.text("Reps : 50")
        if st.button("Like",key=4):
            st.toast('Picture Liked', icon='💓')
        if st.button("Challenge",key=8):
            pushups()