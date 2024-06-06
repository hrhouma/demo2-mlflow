import streamlit as st
import requests

st.title("Application de Machine Learning")

if st.button('Obtenir prédiction'):
    # Appel à l'API FastAPI pour obtenir une prédiction
    response = requests.get('http://localhost:8000/predict')
    prediction = response.json()
    st.write(f"Prédiction: {prediction['prediction']}")  # Affichage de la prédiction

st.write("Ceci est une application Streamlit pour visualiser des modèles de Machine Learning.")
