# app_fastapi.py

from fastapi import FastAPI
import numpy as np
import mlflow.pyfunc

app = FastAPI()

# Charger un modèle MLflow
model_uri = "models:/Iris_Model/Production"
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/predict")
def predict():
    # Simule la réception d'une entrée de données pour la prédiction
    data = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(data)
    return {"message": str(prediction)}
