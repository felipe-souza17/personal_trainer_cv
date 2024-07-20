import streamlit as st
import pandas as pd
from personal_ai import *


st.set_page_config(layout="wide")

personalAI = PersonalAI("legpress_video.mp4")
personalAI.run()

placeholder = st.empty()
st.title("Personal Trainer AI")

while True:
    frame, results = personalAI.image_q.get()

    with placeholder.container():
        st.image(frame)
