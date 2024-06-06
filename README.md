# demo2-mlflow

Voici une version complète et améliorée du README pour votre projet de Machine Learning intégrant Streamlit, FastAPI, et MLflow. Cette version intègre des détails sur l'interaction entre les composants, les scripts à exécuter, ainsi que les instructions pour lancer et visualiser les résultats de l'expérience ML.

### README.md - Projet de Machine Learning avec Streamlit, FastAPI, et MLflow

```markdown
# Projet de Machine Learning avec Streamlit, FastAPI, et MLflow

Ce projet démontre comment intégrer Streamlit, FastAPI et MLflow pour créer une application complète de Machine Learning. L'application permet d'interagir avec un modèle de Machine Learning via une interface utilisateur développée avec Streamlit, gère les requêtes API avec FastAPI et effectue le suivi des expériences avec MLflow.

## Architecture du projet

```
┌────────────────┐           ┌────────────────┐           ┌────────────────┐
│                │           │                │           │                │
│   Frontend     │           │   Backend      │           │   MLOps        │
│ (Streamlit UI) ├──────────►│   (FastAPI)    ├──────────►│   (MLflow)     │
│                │   HTTP    │                │   API     │                │
│                │  Requests │                │  Calls    │                │
└────────────────┘           └────────────────┘           └────────────────┘
```

## Installation

```bash
git clone https://github.com/hrhouma/demo1-mlflow.git
cd demo1-mlflow
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Démarrage des composants

### Démarrer MLflow

Pour suivre les expériences et visualiser les résultats des modèles :

```bash
mlflow ui
```

Accédez à `http://localhost:5000` pour voir l'interface MLflow.

### Démarrer FastAPI

Serveur backend pour gérer les requêtes :

```bash
uvicorn main:app --reload
```

Visitez `http://localhost:8000` pour l'API et `http://localhost:8000/docs` pour la documentation Swagger.

### Démarrer Streamlit

Interface utilisateur pour interagir avec le modèle :

```bash
streamlit run app.py
```

Ouvrez `http://localhost:8501` pour utiliser l'application Streamlit.

## Entraînement du modèle

Pour lancer une expérience de Machine Learning :

```bash
python3 train.py
```

Ce script configure, entraîne un modèle de Machine Learning et enregistre les résultats dans MLflow. Après exécution, vous pouvez visualiser les résultats dans l'interface MLflow.

## Exemples de code pour chaque composant

### Streamlit - Frontend

```python
import streamlit as st
import requests

st.title("Application de Machine Learning")
if st.button('Obtenir prédiction'):
    response = requests.get('http://localhost:8000/predict')
    prediction = response.json()
    st.write(f"Prédiction: {prediction['message']}")
st.write("Ceci est une application Streamlit pour visualiser des modèles de Machine Learning.")
```

### FastAPI - Backend

```python
from fastapi import FastAPI
import numpy as np
import mlflow.pyfunc

app = FastAPI()
model_uri = "models:/monModele/Production"
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/predict")
def predict():
    data = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(data)
    return {"message": str(prediction)}
```

### MLflow - MLOps

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

with mlflow.start_run():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier()
    model.fit(X, y)
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_metric("accuracy", model.score(X, y))
    mlflow.sklearn.log_model(model, "monModele")
```

## Conclusion

Ce guide vous aide à configurer et à utiliser une application de Machine Learning intégrée avec Streamlit, FastAPI, et MLflow pour le suivi des expérimentations. Profitez de cette configuration pour développer, tester et déployer des modèles de Machine Learning de manière efficace et interactive.
```

Ce README fournit une documentation complète pour votre projet, incluant des instructions claires pour l'installation, le démarrage des composants, et l'exécution des scripts. Il offre également une vue d'ensemble claire sur l'architecture et le fonctionnement de votre application.



### Instructions de lancement - Utilisation (3 terminaux)
- **Lancer MLflow** :
```sh
cd demo2-mlflow
source mlflowenv1/bin/activate
streamlit run app.py
```
```sh
cd demo2-mlflow
source mlflowenv1/bin/activate
uvicorn main:app --reload
```
```sh
cd demo1-mlflow
source mlflowenv1/bin/activate
mlflow ui
```
```sh
localhost:8501
localhost:8000 et localhost:8000/docs
localhost:5000
```
Ces exemples améliorent l'interactivité et la fonctionnalité de chaque composant du projet, démontrant une intégration efficace entre le frontend, le backend, et les services MLOps pour une application de Machine Learning complète.
