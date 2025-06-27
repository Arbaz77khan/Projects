# streamlit_app.py

import streamlit as st
import pandas as pd
from inference import Load_model, predict_sms

st.set_page_config(page_title="TextShield", page_icon="🛡️", layout='centered')

st.title("TextShield – Suspicious or Secure? Let's Find Out!")

model = Load_model()

with st.form(key="sms_form"):
    user_input = st.text_area(label = "Drop a message below", height=150)
    submit = st.form_submit_button("Analyze")

if submit:
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        result_df = predict_sms(user_input, model)
        label = result_df['Prediction'].iloc[0]
        prob = result_df['Spam_Probability (%)'].iloc[0]

        if label == "spam":
            st.error("🚨 **SPAM ALERT!** This looks suspicious.")

            st.markdown(f"**Confidence it's spam:** `{prob}%`")
            st.progress(min(int(prob), 100))
        else:
            st.success("✅ **Safe!** This message looks clean.")

st.markdown("---")

