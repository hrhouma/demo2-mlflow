# train_model.py

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration de MLflow
mlflow.set_experiment('my_ml_experiment')

# Démarrer une nouvelle expérience MLflow
with mlflow.start_run():
    # Charger les données
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test =
