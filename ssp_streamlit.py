# ssp_streamlit.py

import streamlit as st
import requests

st.title("Application de Machine Learning Iris")

if st.button('Obtenir prédiction'):
    response = requests.get('http://localhost:8000/predict')
    if response.status_code == 200:
        prediction = response.json()['message']
        st.write(f"Prédiction reçue : {prediction}")
    else:
        st.error("Erreur de connexion avec l'API.")

st.write("Cliquez sur le bouton pour obtenir une prédiction du modèle de Machine Learning.")
