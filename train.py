# train.py

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Créer et entraîner le modèle
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Prédiction et évaluation du modèle
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Logger les paramètres, métriques et le modèle
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Modèle entraîné avec une précision de : {accuracy * 100}%")

