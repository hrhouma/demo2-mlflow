# demo2-mlflow

# Exemple d'architecture - Projet de Machine Learning avec Streamlit, FastAPI, et MLflow


```
/demo1-mlflow
│
├── app_fastapi.py              # Fichier pour l'application Streamlit (frontend)
├── main.py             # Fichier pour l'API FastAPI (backend)
├── train_model.py      # Script pour l'entraînement du modèle et le suivi MLflow
│
├── requirements.txt    # Liste de toutes les dépendances Python nécessaires
├── .gitignore          # Fichier pour ignorer les fichiers/dossiers non souhaités dans Git
│
├── Dockerfile          # (Optionnel) Dockerfile pour la création de l'image Docker
├── docker-compose.yml  # (Optionnel) Fichier pour la configuration de Docker Compose
│
├── README.md           # Documentation du projet
│
├── models              # Dossier pour stocker les modèles entraînés
│   └── monModele       # Dossier pour un modèle spécifique
│
├── notebooks           # Dossier pour les Jupyter notebooks (si utilisés pour l'analyse)
│
├── data                # Dossier pour les jeux de données utilisés dans le projet
│
├── mlruns              # Dossier généré automatiquement par MLflow pour stocker les runs
│
├── scripts             # Dossier pour les scripts supplémentaires utilisés dans le projet
│
└── tests               # Dossier pour les tests unitaires et d'intégration
```

### Explications des composants clés :
- **app.py** : Ce fichier contient le code pour l'interface utilisateur Streamlit. Il interagit avec le backend via des appels API et affiche les résultats des prédictions.
- **main.py** : Ce fichier définit le backend utilisant FastAPI. Il expose des endpoints pour les opérations de Machine Learning et communique avec MLflow pour récupérer les modèles.
- **train.py** : Script pour entraîner les modèles de Machine Learning et enregistrer les résultats dans MLflow.
- **requirements.txt** : Contient toutes les bibliothèques Python nécessaires à l'installation pour que le projet fonctionne correctement.
- **Dockerfile** et **docker-compose.yml** : Fournissent les configurations nécessaires pour dockeriser l'application, permettant une déploiement facile et une meilleure portabilité.
- **README.md** : Documentation détaillée sur le projet, son installation, son utilisation et son architecture.
- **models** : Dossier pour stocker les modèles MLflow enregistrés ou tout autre modèle de Machine Learning.
- **notebooks** : Pour les analyses exploratoires de données ou tout autre traitement préliminaire des données avec Jupyter Notebooks.
- **data** : Pour stocker les jeux de données utilisés dans vos expériences de Machine Learning.
- **mlruns** : Répertoire généré par MLflow pour stocker les informations sur les différents runs et expériences.
- **scripts** : Contient tout script supplémentaire qui pourrait être utilisé pour le prétraitement des données, des analyses supplémentaires, etc.
- **tests** : Pour les tests unitaires et d'intégration qui garantissent la robustesse de votre code.


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
git clone https://github.com/hrhouma/demo2-mlflow.git
cd demo2-mlflow
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Instructions de lancement - Utilisation (3 terminaux)



1. **Ouvrez un terminal** et installez Python 3 et pip si ce n'est pas déjà fait :
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```

2. **Clonez le dépôt et configurez l'environnement virtuel** :
   ```bash
   git clone https://github.com/hrhouma/demo2-mlflow.git
   cd demo1-mlflow
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

# Étape 1 : Démarrer les Composants


# TERMINAL 1
```sh
cd demo1-mlflow
source mlflowenv1/bin/activate
python3 train_model.py
mlflow ui
```
1. **Entraîner le modèle**:
   ```bash
   python3 train_model.py
   ```
   Cette commande va entraîner votre modèle sur le dataset Iris, enregistrer le modèle dans MLflow, et configurer la gestion des versions du modèle dans le registre MLflow.

2. **Lancer MLflow UI**:
   ```bash
   mlflow ui
   ```
   Cette commande démarre l'interface utilisateur MLflow sur `http://127.0.0.1:5000`, où vous pouvez visualiser les détails des expériences, y compris les métriques et les versions du modèle.


# TERMINAL 2
```sh
cd demo2-mlflow
source mlflowenv1/bin/activate
uvicorn app_fastapi:app --reload
```
3. **Démarrer le serveur FastAPI**:
   ```bash
   uvicorn app_fastapi:app --reload
   ```
   Cette commande démarre le serveur FastAPI sur `http://127.0.0.1:8000`. Vous pouvez accéder à Swagger UI pour tester l'API à `http://127.0.0.1:8000/docs` et essayer l'endpoint `/predict` pour voir les prédictions en temps réel.



# TERMINAL 3
- **Lancer MLflow** :
```sh
cd demo2-mlflow
source mlflowenv1/bin/activate
streamlit run ssp_streamlit.py
```
4. **Exécuter l'application Streamlit**:
   ```bash
   streamlit run ssp_streamlit.py
   ```
   Lance l'application Streamlit qui se connecte au backend FastAPI pour obtenir des prédictions. L'interface utilisateur Streamlit sera disponible sur `http://localhost:8501`.

### Actions et URL à visiter:

- **MLflow UI**: Visitez `http://127.0.0.1:5000` pour visualiser les détails des runs et la gestion des modèles.
- **API FastAPI**: Allez sur `http://127.0.0.1:8000` pour voir l'API en action. Utilisez `http://127.0.0.1:8000/docs` pour interagir avec Swagger UI et tester l'endpoint `/predict`.
- **Streamlit UI**: Ouvrez `http://localhost:8501` et utilisez l'interface Streamlit pour obtenir des prédictions en cliquant sur le bouton prévu à cet effet.

# Résumé des adresses
```sh
localhost:8501
localhost:8000 et localhost:8000/docs
localhost:5000
```
Assurez-vous que chaque service soit lancé dans un terminal distinct pour pouvoir les exécuter simultanément sans interruption.

# Annexe 1 - Code final


### 1. Fichier FastAPI (`app_fastapi.py`)

```python
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
```

### 2. Fichier Streamlit (`ssp_streamlit.py`)

```python
import streamlit as st
import requests

st.title("Application de Machine Learning")

if st.button('Obtenir prédiction'):
    # Appel à l'API FastAPI pour obtenir une prédiction
    response = requests.get('http://localhost:8000/predict')
    prediction = response.json()
    st.write(f"Prédiction: {prediction['prediction']}")  # Affichage de la prédiction

st.write("Ceci est une application Streamlit pour visualiser des modèles de Machine Learning.")
```

### 3. Script de formation et enregistrement du modèle (`train_model.py`)

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration de MLflow pour utiliser une expérience spécifique
mlflow.set_experiment('Iris_Model_Experiment')

with mlflow.start_run():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Enregistrement des métriques, des paramètres et du modèle
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model", registered_model_name="Iris_Model")
    
    # Gestion des stages du modèle
    client = mlflow.tracking.MlflowClient()
    model_versions = client.search_model_versions(f"name='Iris_Model'")
    for mv in model_versions:
        if mv.current_stage == 'Production':
            client.transition_model_version_stage(
                name="Iris_Model",
                version=mv.version,
                stage="Archived"
            )

    client.transition_model_version_stage(
        name="Iris_Model",
        version=model_versions[0].version,
        stage="Production"
    )

    print(f"Modèle entraîné avec une précision de : {accuracy * 100}%")
```

### Instructions pour l'exécution:
1. **Exécuter le script de formation du modèle** (`train_model.py`) pour entraîner et enregistrer le modèle dans MLflow.
2. **Démarrer le serveur FastAPI** avec `uvicorn app_fastapi:app --reload` pour servir le modèle.
3. **Ouvrir l'application Streamlit** avec `streamlit run ssp_streamlit.py` pour interagir avec le modèle via l'interface utilisateur.

Assurez-vous que tous les environnements et dépendances nécessaires sont correctement configurés avant de lancer les scripts.

# Annexe 2 - Explications supplémeantaires : 
- Si vous obtenez `"Prédiction: [0]"` ou une sortie similaire avec un "0", cela indique la prédiction faite par votre modèle de machine learning. Dans le contexte de l'exemple que nous avons utilisé, qui utilise le dataset Iris avec un modèle de RandomForestClassifier, ce "0" représente la classe prédite pour l'entrée donnée.

Le dataset Iris comprend trois classes de fleurs iris:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

Ces classes sont typiquement encodées en tant que 0, 1, et 2. Donc, une prédiction de "0" signifierait que le modèle prédit que l'échantillon entré appartient à la classe "Iris Setosa".

### Pourquoi "[0]"?
Le format "[0]" est une liste contenant un seul élément, qui est 0. Cela se produit parce que de nombreux modèles de scikit-learn, y compris RandomForestClassifier, retournent des prédictions sous forme de tableau numpy. Quand vous convertissez ce tableau en liste pour le rendre compatible JSON (comme dans votre API FastAPI), il apparaît sous forme de liste.

Pour améliorer la clarté de l'interface utilisateur, vous pouvez modifier le code pour afficher un message plus descriptif en fonction de la prédiction, par exemple :

```python
# Modification dans app_fastapi.py pour inclure les noms des classes
@app.get("/predict")
def predict():
    data = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(data)
    classes = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]  # Ajout des noms de classes
    predicted_class = classes[prediction[0]]  # Obtention du nom de la classe prédite
    return {"prediction": predicted_class}
```

Et ajustez votre application Streamlit pour traiter cette réponse :

```python
# Modification dans ssp_streamlit.py pour afficher le nom de la classe
if st.button('Obtenir prédiction'):
    response = requests.get('http://localhost:8000/predict')
    prediction = response.json()
    st.write(f"Prédiction: {prediction['prediction']}")
```

Ces modifications aideront à rendre les résultats plus compréhensibles pour les utilisateurs finaux en affichant directement le nom de la classe prédite au lieu de son encodage numérique.
