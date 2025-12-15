"""
Model training and tracking with MLflow.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# Custom imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import create_feature_pipeline

warnings.filterwarnings('ignore')


class CreditRiskModelTrainer:
    """Train and track credit risk models."""
    
    def __init__(self, 
                 experiment_name: str = "credit_risk_modeling",
                 tracking_uri: str = "sqlite:///mlflow.db",
                 random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
            random_state: Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.random_state = random_state
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Initialize models dictionary
        self.models = {
            'logistic_regression': LogisticRegression(random_state=random_state),
            'decision_tree': DecisionTreeClassifier(random_state=random_state),
            'random_forest': RandomForestClassifier(random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state)
        }
        
        # Define hyperparameter grids for each model
        self.param_grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 200, 500]
            },
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def prepare_data(self, 
                     data_path: str,
                     target_col: str = 'is_high_risk',
                     test_size: float = 0.2,
                     val_size: float = 0.1) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            data_path: Path to processed data
            target_col: Name of target column
            test_size: Proportion for test set
            val_size: Proportion for validation set (from training set)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Load data
        data = pd.read_csv(data_path)
        
        # Separate features and target
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Handle CustomerId if present
        if 'CustomerId' in X.columns:
            X = X.drop(columns=['CustomerId'])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Train-validation split
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_relative_size,
            random_state=self.random_state,
            stratify=y_train
        )
        
        print(f"Data shapes:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_model(self, 
                       model, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       set_name: str = "validation") -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            set_name: Name of dataset (for logging)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            f'{set_name}_accuracy': accuracy_score(y, y_pred),
            f'{set_name}_precision': precision_score(y, y_pred, zero_division=0),
            f'{set_name}_recall': recall_score(y, y_pred, zero_division=0),
            f'{set_name}_f1': f1_score(y, y_pred, zero_division=0),
        }
        
        # Add ROC-AUC if probability predictions are available
        if y_pred_proba is not None:
            metrics[f'{set_name}_roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        # Add confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics[f'{set_name}_tn'] = int(cm[0, 0])
        metrics[f'{set_name}_fp'] = int(cm[0, 1])
        metrics[f'{set_name}_fn'] = int(cm[1, 0])
        metrics[f'{set_name}_tp'] = int(cm[1, 1])
        
        return metrics
    
    def train_model(self, 
                    model_name: str,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: pd.DataFrame,
                    y_val: pd.Series,
                    use_grid_search: bool = True,
                    cv_folds: int = 5) -> Tuple:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            use_grid_search: Whether to use GridSearchCV (True) or RandomizedSearchCV (False)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (trained model, best parameters, validation metrics)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not supported")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("use_grid_search", use_grid_search)
            mlflow.log_param("cv_folds", cv_folds)
            
            # Get base model and parameter grid
            model = self.models[model_name]
            param_grid = self.param_grids[model_name]
            
            # Perform hyperparameter tuning
            if use_grid_search:
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
            else:
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=20,  # Number of parameter settings sampled
                    cv=cv_folds,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1,
                    random_state=self.random_state
                )
            
            # Train with cross-validation
            print(f"\nTraining {model_name} with {'Grid' if use_grid_search else 'Randomized'} Search...")
            search.fit(X_train, y_train)
            
            # Get best model
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_
            
            # Log hyperparameters
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            mlflow.log_metric("best_cv_score", best_score)
            
            # Evaluate on validation set
            val_metrics = self.evaluate_model(best_model, X_val, y_val, "val")
            
            # Log validation metrics
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=f"{model_name}_model",
                signature=infer_signature(X_val, best_model.predict(X_val))
            )
            
            # Log feature importance for tree-based models
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance as CSV artifact
                importance_path = f"feature_importance_{model_name}.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)  # Clean up
            
            print(f"Best parameters: {best_params}")
            print(f"Best CV score: {best_score:.4f}")
            print(f"Validation ROC-AUC: {val_metrics.get('val_roc_auc', 'N/A'):.4f}")
            
            return best_model, best_params, val_metrics
    
    def train_all_models(self, 
                         X_train: pd.DataFrame,
                         y_train: pd.Series,
                         X_val: pd.DataFrame,
                         y_val: pd.Series,
                         models_to_train: List[str] = None) -> Dict:
        """
        Train multiple models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            models_to_train: List of model names to train
            
        Returns:
            Dictionary of trained models and their metrics
        """
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'random_forest']
        
        trained_models = {}
        
        for model_name in models_to_train:
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print(f"{'='*50}")
            
            try:
                model, params, metrics = self.train_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    use_grid_search=True
                )
                
                trained_models[model_name] = {
                    'model': model,
                    'params': params,
                    'metrics': metrics
                }
                
                # Update best model
                roc_auc = metrics.get('val_roc_auc', 0)
                if roc_auc > self.best_score:
                    self.best_score = roc_auc
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        print(f"\n{'='*50}")
        print(f"Training Summary")
        print(f"{'='*50}")
        for model_name, results in trained_models.items():
            print(f"{model_name}: ROC-AUC = {results['metrics'].get('val_roc_auc', 'N/A'):.4f}")
        
        print(f"\nBest model: {self.best_model_name} (ROC-AUC: {self.best_score:.4f})")
        
        return trained_models
    
    def evaluate_on_test(self, 
                         X_test: pd.DataFrame,
                         y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate best model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Test metrics
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        test_metrics = self.evaluate_model(self.best_model, X_test, y_test, "test")
        
        # Log test metrics in a new run
        with mlflow.start_run(run_name=f"test_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("best_model", self.best_model_name)
            mlflow.log_metric("best_val_score", self.best_score)
            
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log best model to registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "credit_risk_model")
        
        return test_metrics
    
    def save_best_model(self, 
                        model_path: str = "models/best_model.pkl",
                        metadata_path: str = "models/model_metadata.json"):
        """
        Save the best model and its metadata.
        
        Args:
            model_path: Path to save the model
            metadata_path: Path to save model metadata
        """
        import joblib
        import json
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'best_score': self.best_score,
            'training_date': datetime.now().isoformat(),
            'random_state': self.random_state,
            'model_type': type(self.best_model).__name__
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Best model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")


def main():
    """Main training script."""
    # Configuration
    config = {
        'data_path': 'data/processed/final_dataset.csv',
        'target_col': 'is_high_risk',
        'test_size': 0.2,
        'val_size': 0.1,
        'models_to_train': ['logistic_regression', 'random_forest', 'gradient_boosting'],
        'experiment_name': 'credit_risk_v1',
        'random_state': 42
    }
    
    # Initialize trainer
    trainer = CreditRiskModelTrainer(
        experiment_name=config['experiment_name'],
        random_state=config['random_state']
    )
    
    # Prepare data
    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        data_path=config['data_path'],
        target_col=config['target_col'],
        test_size=config['test_size'],
        val_size=config['val_size']
    )
    
    # Train models
    print("\nTraining models...")
    trained_models = trainer.train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        models_to_train=config['models_to_train']
    )
    
    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    test_metrics = trainer.evaluate_on_test(X_test, y_test)
    
    print(f"\nTest Metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save best model
    trainer.save_best_model(
        model_path="models/best_model.pkl",
        metadata_path="models/model_metadata.json"
    )
    
    # Generate classification report
    y_pred = trainer.best_model.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"[[TN={cm[0,0]} FP={cm[0,1]}]")
    print(f" [FN={cm[1,0]} TP={cm[1,1]}]]")


if __name__ == "__main__":
    main()