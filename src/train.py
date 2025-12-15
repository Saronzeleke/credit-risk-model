# src/train.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os
from datetime import datetime

# Import custom modules
from src.data_processing import process_data, create_data_pipeline
from src.target_engineering import RFMTargetEngineer

def load_and_prepare_data(data_path):
    """Load data and engineer target variable"""
    # Load raw data
    df = pd.read_csv(data_path)
    
    # Engineer target variable
    target_engineer = RFMTargetEngineer(random_state=42)
    df_with_target = target_engineer.engineer_target_variable(df)
    
    # Separate features and target
    X = df_with_target.drop(columns=['is_high_risk'])
    y = df_with_target['is_high_risk']
    
    return X, y

def train_models(X, y, experiment_name="credit_risk_modeling"):
    """Train and track multiple models"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:///mlruns")
    
    # Create experiment
    mlflow.set_experiment(experiment_name)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create data pipeline
    pipeline = create_data_pipeline()
    
    # Fit pipeline on training data
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)
    
    # Define models and hyperparameters
    models = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    }
    
    best_models = {}
    
    for model_name, model_config in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print(f"\nTraining {model_name}...")
            
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            # Train model
            grid_search.fit(X_train_processed, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test_processed)
            y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log parameters
            mlflow.log_params(grid_search.best_params_)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(best_model, model_name)
            
            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_dict(report, f"{model_name}_classification_report.json")
            
            # Save best model
            best_models[model_name] = {
                'model': best_model,
                'metrics': metrics,
                'params': grid_search.best_params_
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Metrics: {metrics}")
    
    return best_models, pipeline

def register_best_model(best_models, pipeline):
    """Register the best model in MLflow registry"""
    
    # Find best model based on ROC-AUC
    best_model_name = None
    best_roc_auc = 0
    
    for model_name, model_info in best_models.items():
        if model_info['metrics']['roc_auc'] > best_roc_auc:
            best_roc_auc = model_info['metrics']['roc_auc']
            best_model_name = model_name
    
    if best_model_name:
        print(f"\nBest model: {best_model_name} with ROC-AUC: {best_roc_auc}")
        
        # Register the best model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model_name}"
        registered_model = mlflow.register_model(model_uri, "credit_risk_model")
        
        # Save pipeline
        joblib.dump(pipeline, 'models/data_pipeline.pkl')
        
        return registered_model
    
    return None

def main():
    """Main training function"""
    
    # Data path
    data_path = "data/raw/transactions.csv"
    
    # Load and prepare data
    print("Loading and preparing data...")
    X, y = load_and_prepare_data(data_path)
    
    # Train models
    print("\nTraining models...")
    best_models, pipeline = train_models(X, y)
    
    # Register best model
    print("\nRegistering best model...")
    registered_model = register_best_model(best_models, pipeline)
    
    if registered_model:
        print(f"Model registered: {registered_model.name} version {registered_model.version}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()