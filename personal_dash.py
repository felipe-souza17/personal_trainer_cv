import streamlit as st
import pandas as pd
from personal_ai import *


st.set_page_config(layout="wide")

personalAI = PersonalAI("terra.mp4")
personalAI.run()

status = "descansando"
count = 0
st.title("Personal Trainer AI")
placeholder = st.empty()

while True:
    frame, results,ts = personalAI.image_q.get()
    if ts == "done": break

    if len(results.pose_landmarks) > 0:
        hip_angle = personalAI.find_angle(results, 24, 26, 28)
        elbow_angle = personalAI.find_angle(results, 12, 24, 26)

        # Lógica do terra
        if hip_angle > 170 and elbow_angle < 60:
            status = "pronto"
            dir = "down"

        if status == "pronto":
            if dir == "down" and elbow_angle > 100:
                dir = "up"
                count += 0.5
            if dir == "up" and elbow_angle < 60:
                dir = "down"
                count += 0.5

    with placeholder.container():
        col1, col2 = st.columns([0.4, 0.6])
        col1.image(frame)
        col2.markdown("## **Status:** " + status)
        col2.markdown(f"### Repetições: {int(count)}")
