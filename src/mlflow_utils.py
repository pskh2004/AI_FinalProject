# src/mlflow_utils.py
import os
import mlflow
import mlflow.sklearn

# Switch Matplotlib to 'Agg' backend to prevent Tkinter/Tcl errors
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix


def init_mlflow_experiment(experiment_name="Customer_Churn_Prediction"):
    """
    Initializes or sets the active MLflows experiments.
    """
    mlflow.set_experiment(experiment_name)


def log_experiment_run(run_name, model_name, params, metrics, data_version, y_true, y_pred, model=None):
    """
    Logs all metrics, parameters, confusion matrix, and model artifacts to MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_version", data_version)

        for p_name, p_val in params.items():
            mlflow.log_param(p_name, p_val)

        for m_name, m_val in metrics.items():
            mlflow.log_metric(m_name, m_val)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)
        if os.path.exists(cm_path):
            os.remove(cm_path)

        # Forces MLflow to use traditional 'pickle' format to trust XGBoost/CatBoost classes
        if model is not None:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                serialization_format="pickle"
            )

        print(f"[+] Run '{run_name}' with version '{data_version}' logged into MLflow.")