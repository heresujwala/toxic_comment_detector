import streamlit as st
import requests

st.title("Toxic Comment Classifier")
user_input = st.text_area("Enter a comment:")

if st.button("Classify"):
    response = requests.post("http://localhost:8000/predict", json={"text": user_input})
    result = response.json()
    st.write("Toxicity:", "Yes" if result["prediction"] else "No")
    st.write("Confidence:", round(result["probability"] * 100, 2), "%")

    feedback = st.radio("Is this prediction correct?", ["Yes", "No"])
    if feedback == "No":
        st.write("Thanks! We'll use your correction for future model improvements.")
