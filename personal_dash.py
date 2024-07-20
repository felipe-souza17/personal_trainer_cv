import streamlit as st
import pandas as pd
from personal_ai import *


st.set_page_config(layout="wide")

personalAI = PersonalAI("legpress_video.mp4")

st.title("Personal Trainer AI")