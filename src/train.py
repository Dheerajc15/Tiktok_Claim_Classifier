import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Allow imports from src/ when running as a script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import prepare_data


# ----------------------------- Config ----------------------------- #

EXPERIMENT_NAME = "tiktok_claim_classifier"
REGISTERED_MODEL_NAME = "tiktok_claim_classifier_rf"
DATA_PATH = "data/tiktok_dataset.csv"
TRACKING_URI = "sqlite:///mlflow.db"

RF_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

XGB_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [4, 6],
    "learning_rate": [0.1, 0.2],
}


# ----------------------------- Helpers ----------------------------- #

def evaluate(model, X, y, split_name: str) -> dict:
    """Compute classification metrics and return as a dict."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        f"{split_name}_accuracy": accuracy_score(y, preds),
        f"{split_name}_precision": precision_score(y, preds),
        f"{split_name}_recall": recall_score(y, preds),
        f"{split_name}_f1": f1_score(y, preds),
    }
    if probs is not None:
        metrics[f"{split_name}_roc_auc"] = roc_auc_score(y, probs)

    return metrics


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train RF with GridSearchCV, log everything to MLflow."""
    with mlflow.start_run(run_name="random_forest") as run:
        mlflow.set_tag("model_family", "random_forest")
        mlflow.set_tag("stage", "experiment")

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid = GridSearchCV(
            rf,
            param_grid=RF_PARAM_GRID,
            cv=5,
            scoring="recall",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # Log best hyperparameters
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "recall")

        # Log metrics across splits
        val_metrics = evaluate(best_model, X_val, y_val, "val")
        test_metrics = evaluate(best_model, X_test, y_test, "test")
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("cv_best_recall", grid.best_score_)

        # Log feature importances as artifact
        importances = pd.DataFrame(
            {"feature": X_train.columns, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        importances.to_csv("rf_feature_importances.csv", index=False)
        mlflow.log_artifact("rf_feature_importances.csv")
        os.remove("rf_feature_importances.csv")

        # Log model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train.head(2),
        )

        print(f"\n[RF] Best params: {grid.best_params_}")
        print(f"[RF] Val metrics:  {val_metrics}")
        print(f"[RF] Test metrics: {test_metrics}")

        return best_model, test_metrics, run.info.run_id


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost with GridSearchCV, log everything to MLflow."""
    with mlflow.start_run(run_name="xgboost") as run:
        mlflow.set_tag("model_family", "xgboost")
        mlflow.set_tag("stage", "experiment")

        xgb = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        grid = GridSearchCV(
            xgb,
            param_grid=XGB_PARAM_GRID,
            cv=5,
            scoring="recall",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "recall")

        val_metrics = evaluate(best_model, X_val, y_val, "val")
        test_metrics = evaluate(best_model, X_test, y_test, "test")
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("cv_best_recall", grid.best_score_)

        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model",
            input_example=X_train.head(2),
        )

        print(f"\n[XGB] Best params: {grid.best_params_}")
        print(f"[XGB] Val metrics:  {val_metrics}")
        print(f"[XGB] Test metrics: {test_metrics}")

        return best_model, test_metrics, run.info.run_id


def register_champion(run_id: str, model_name: str = REGISTERED_MODEL_NAME):
    """Register the champion model in MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"\n[Registry] Registered {model_name} version {result.version}")

    # Transition to Production stage (MLflow legacy stages — still widely used)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[Registry] Transitioned version {result.version} to Production")


# ----------------------------- Main ----------------------------- #

def main(data_path: str):
    # Setup MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Loading and preprocessing data from {data_path} ...")
    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_data(data_path)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train both models
    rf_model, rf_test_metrics, rf_run_id = train_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    xgb_model, xgb_test_metrics, xgb_run_id = train_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Champion selection: Random Forest (per notebook conclusion)
    # Even if XGBoost ties on recall, RF wins on interpretability + speed
    print("\n" + "=" * 60)
    print("Registering Random Forest as champion model")
    print("=" * 60)
    register_champion(rf_run_id)

    print("\nDone. Launch the MLflow UI with:")
    print("    mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_PATH,
        help="Path to tiktok_dataset.csv",
    )
    args = parser.parse_args()
    main(args.data_path)