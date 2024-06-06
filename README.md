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
