import streamlit as st
import requests

st.title("Application de Machine Learning")

# Interaction avec le backend FastAPI
if st.button('Obtenir prédiction'):
    response = requests.get('http://localhost:8000/predict')
    prediction = response.json()
    st.write(f"Prédiction: {prediction['message']}")

st.write("Ceci est une application Streamlit pour visualiser des modèles de Machine Learning.")

