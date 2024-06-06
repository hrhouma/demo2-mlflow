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
