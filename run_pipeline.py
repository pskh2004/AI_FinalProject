# run_pipeline.py
import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings to keep console output clean
warnings.filterwarnings('ignore', category=ConvergenceWarning)

from src.data_loader import load_raw_data, save_data_version
from src.preprocessing import clean_data
from src.features import engineer_features
from src.train import get_models_and_grids, train_and_tune_model
from src.evaluate import evaluate_model
from src.mlflow_utils import init_mlflow_experiment, log_experiment_run


def main():
    data_dir = "data"
    os.makedirs(os.path.join(data_dir, "v1"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "v2"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "v3"), exist_ok=True)

    raw_path = os.path.join(data_dir, "v1", "telco_customer_churn.csv")
    if not os.path.exists(raw_path):
        print(f"[-] Error: Raw dataset not found at {raw_path}.")
        print("Please place the IBM Telco Customer Churn dataset named 'telco_customer_churn.csv' in 'data/v1/'.")
        return

    # 1. Load Raw Data (v1)
    print("[+] Loading Raw Data (v1)...")
    df_v1 = load_raw_data(raw_path)

    # 2. Preprocess and Save Cleaned Data (v2)
    print("[+] Preprocessing v2 (Data Cleaning)...")
    df_v2 = clean_data(df_v1)
    save_data_version(df_v2, os.path.join(data_dir, "v2", "telco_customer_churn_clean.csv"))

    # 3. Feature Engineering and Save (v3)
    print("[+] Feature Engineering v3...")
    df_v3 = engineer_features(df_v2)
    save_data_version(df_v3, os.path.join(data_dir, "v3", "telco_customer_churn_features.csv"))

    # 4. Initialize MLflow
    init_mlflow_experiment("Customer_Churn_Prediction")

    datasets = {
        "v2": df_v2,
        "v3": df_v3
    }

    seed = 42

    for version_name, df_version in datasets.items():
        print(f"\n=========================================")
        print(f"Training and Evaluating on Data Version: {version_name}")
        print(f"=========================================")

        if 'Churn Value' not in df_version.columns:
            print(f"[-] Error: 'Churn Value' column not found in data version {version_name}.")
            continue

        X = df_version.drop(columns=['Churn Value'])
        y = df_version['Churn Value']

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        models, grids = get_models_and_grids(seed=seed)

        for model_name, model in models.items():
            print(f"-> Training and tuning {model_name}...")

            # Hyperparameter tuning with cross-validation
            best_model, best_params = train_and_tune_model(
                X_train, y_train, model_name, model, grids[model_name], seed=seed
            )

            # Evaluation
            metrics, y_pred = evaluate_model(best_model, X_test, y_test)

            # Log run to MLflow
            run_name = f"{model_name.replace(' ', '_')}_{version_name}"
            log_experiment_run(
                run_name=run_name,
                model_name=model_name,
                params=best_params,
                metrics=metrics,
                data_version=version_name,
                y_true=y_test,
                y_pred=y_pred,
                model=best_model
            )

            print(f"    Test Metrics: {metrics}")

    print("\n[+] Pipeline execution completed successfully. Results are logged in MLflow.")


if __name__ == "__main__":
    main()