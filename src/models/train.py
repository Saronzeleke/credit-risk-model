import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve)
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import yaml
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.data.preprocess import preprocess_and_split


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_models(
    raw_data_path: str = "data/raw/transactions.csv",
    target_column: str = "is_high_risk",
    config_path: str = "configs/config.yaml"
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Train and evaluate models with MLflow tracking.
    
    Returns:
        models_results: Dict with model names and (model, metrics) tuples
        feature_names: List of feature names
    """
    cfg = load_config(config_path)
    mlflow_cfg = cfg['mlflow']
    
    # Set up MLflow
    mlflow.set_experiment(mlflow_cfg['experiment_name'])
    mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, feature_names, pipeline = preprocess_and_split(
        raw_data_path, target_column, config_path
    )

    models_results = {}

    with mlflow.start_run() as run:
        # Log preprocessing info
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Logistic Regression (interpretable baseline)
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(
            random_state=cfg['pipeline']['random_state'],
            **cfg['models']['logistic']
        )
        lr_grid = GridSearchCV(
            lr, 
            {'C': [0.01, 0.1, 1, 10]}, 
            cv=5, 
            scoring='roc_auc'
        )
        lr_grid.fit(X_train, y_train)
        
        # Evaluate
        lr_pred_proba = lr_grid.predict_proba(X_test)[:, 1]
        lr_pred = lr_grid.predict(X_test)
        metrics_lr = {
            'auc': roc_auc_score(y_test, lr_pred_proba),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'f1': f1_score(y_test, lr_pred)
        }
        
        # Log metrics
        mlflow.log_metrics({f"lr_{k}": v for k, v in metrics_lr.items()})
        mlflow.log_param("lr_best_C", lr_grid.best_params_['C'])
        
        # Save model
        mlflow.sklearn.log_model(lr_grid, "logistic_model")
        joblib.dump(lr_grid, "models/logistic_model.joblib")
        models_results['logistic'] = (lr_grid, metrics_lr)
        
        print(f"Logistic Regression AUC: {metrics_lr['auc']:.4f}")

        # Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(
            random_state=cfg['pipeline']['random_state']
        )
        rf_grid = GridSearchCV(
            rf,
            cfg['models']['random_forest'],
            cv=5,
            scoring='roc_auc'
        )
        rf_grid.fit(X_train, y_train)
        
        rf_pred_proba = rf_grid.predict_proba(X_test)[:, 1]
        rf_pred = rf_grid.predict(X_test)
        metrics_rf = {
            'auc': roc_auc_score(y_test, rf_pred_proba),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred)
        }
        
        mlflow.log_metrics({f"rf_{k}": v for k, v in metrics_rf.items()})
        mlflow.sklearn.log_model(rf_grid, "rf_model")
        joblib.dump(rf_grid, "models/rf_model.joblib")
        models_results['rf'] = (rf_grid, metrics_rf)
        
        print(f"Random Forest AUC: {metrics_rf['auc']:.4f}")

        # Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb = GradientBoostingClassifier(
            random_state=cfg['pipeline']['random_state']
        )
        gb_grid = GridSearchCV(
            gb,
            cfg['models']['gradient_boosting'],
            cv=5,
            scoring='roc_auc'
        )
        gb_grid.fit(X_train, y_train)
        
        gb_pred_proba = gb_grid.predict_proba(X_test)[:, 1]
        gb_pred = gb_grid.predict(X_test)
        metrics_gb = {
            'auc': roc_auc_score(y_test, gb_pred_proba),
            'precision': precision_score(y_test, gb_pred),
            'recall': recall_score(y_test, gb_pred),
            'f1': f1_score(y_test, gb_pred)
        }
        
        mlflow.log_metrics({f"gb_{k}": v for k, v in metrics_gb.items()})
        mlflow.sklearn.log_model(gb_grid, "gb_model")
        joblib.dump(gb_grid, "models/gb_model.joblib")
        models_results['gb'] = (gb_grid, metrics_gb)
        
        print(f"Gradient Boosting AUC: {metrics_gb['auc']:.4f}")

        # Save best model and pipeline
        best_model_name = max(models_results, key=lambda k: models_results[k][1]['auc'])
        best_model = models_results[best_model_name][0]
        joblib.dump(best_model, "models/best_model.joblib")
        joblib.dump(pipeline, "models/pipeline.joblib")
        
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_auc", models_results[best_model_name][1]['auc'])
        
        # Generate and save ROC curve
        plt.figure(figsize=(8, 6))
        for name, (model, _) in models_results.items():
            pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, pred_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC={metrics_lr['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig('reports/roc_curves.png')
        plt.close()
        
        print(f"\nBest model: {best_model_name} with AUC: {models_results[best_model_name][1]['auc']:.4f}")

    return models_results, feature_names


if __name__ == "__main__":
    results, features = train_models()
    print("\nTraining complete!")
    print("\nFinal Results:")
    for name, (_, metrics) in results.items():
        print(f"{name}: AUC={metrics['auc']:.4f}, Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")