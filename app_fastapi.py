from fastapi import FastAPI
import numpy as np
import mlflow.pyfunc

app = FastAPI()
model_uri = "models:/Iris_Model/Production"
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/predict")
def predict():
    # Exemple de prédiction avec des valeurs fixes
    data = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}  # Convertir en liste pour la compatibilité JSON
