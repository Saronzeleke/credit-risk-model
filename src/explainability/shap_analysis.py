"""
SHAP Analysis for Credit Risk Model
Professional implementation using processed test data
"""

import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import os
from typing import Any, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = r"C:\Users\admin\credit-risk-model\models"
PROCESSED_DATA_DIR = r"C:\Users\admin\credit-risk-model\data\processed"
REPORTS_DIR = r"C:\Users\admin\credit-risk-model\reports"

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_test_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load processed test data and feature names.
    
    Returns:
        X_test: Test features (numeric only)
        y_test: Test labels
        feature_names: List of feature names
    """
    try:
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_test.csv")).squeeze()
        
        # 🔥 FIX: Keep only numeric columns (drop ID columns)
        id_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
                      'CustomerId', 'ProductId']
        
        # Drop ID columns if they exist
        cols_to_drop = [col for col in id_columns if col in X_test.columns]
        if cols_to_drop:
            X_test = X_test.drop(columns=cols_to_drop)
            logger.info(f"Dropped ID columns: {cols_to_drop}")
        
        # Keep only numeric columns
        X_test = X_test.select_dtypes(include=[np.number])
        
        # Get feature names
        feature_names = X_test.columns.tolist()
        
        logger.info(f"✅ Loaded test data: {X_test.shape}")
        logger.info(f"✅ Numeric features: {len(feature_names)}")
        logger.info(f"✅ Features: {feature_names[:10]}...")  # Show first 10
        
        return X_test, y_test, feature_names
        
    except FileNotFoundError as e:
        logger.error(f"❌ Test data not found: {e}")
        logger.info("Please ensure X_test.csv and y_test.csv exist in data/processed/")
        raise


def load_model(model_name: str = "xgboost") -> Any:
    """
    Load trained model by name.
    
    Args:
        model_name: 'xgboost', 'random_forest', 'gradient_boosting', or 'logistic'
    
    Returns:
        Trained model
    """
    model_files = {
        'xgboost': os.path.join(MODELS_DIR, "xgboost_model.pkl"),
        'random_forest': os.path.join(MODELS_DIR, "random_forest_model.pkl"),
        'gradient_boosting': os.path.join(MODELS_DIR, "gradient_boosting_model.pkl"),
        'logistic': os.path.join(MODELS_DIR, "logistic_model.pkl")
    }
    
    if model_name not in model_files:
        raise ValueError(f"Model must be one of: {list(model_files.keys())}")
    
    model_path = model_files[model_name]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_data = joblib.load(model_path)
    
    # Handle different save formats
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model']
        logger.info(f"Loaded {model_name} with metrics: {model_data.get('metrics', {})}")
    else:
        model = model_data
    
    logger.info(f"✅ Loaded model: {model_name}")
    return model


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    sample_size: int = 500
) -> Tuple[Any, np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values with validation.
    
    Args:
        model: Trained model
        X: Feature matrix (numeric only)
        sample_size: Number of samples to use
    
    Returns:
        explainer: SHAP explainer
        shap_values: Computed SHAP values
        X_sample: Sampled data used
    """
    # 🔥 FIX: Ensure X is numeric
    X = X.select_dtypes(include=[np.number])
    
    # Sample data for efficiency
    X_sample = X.sample(min(sample_size, len(X)), random_state=42)
    
    # 🔥 FIX: Convert to float32 for XGBoost
    X_sample_float = X_sample.astype(np.float32)
    
    logger.info(f"Computing SHAP values on {len(X_sample)} samples...")
    
    try:
        # Choose appropriate explainer based on model type
        if hasattr(model, 'coef_'):  # Linear models
            explainer = shap.LinearExplainer(model, X_sample_float)
            shap_values = explainer.shap_values(X_sample_float)
            
        elif hasattr(model, 'estimators_'):  # Tree-based (Random Forest, Gradient Boosting)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_float)
            
        elif 'xgboost' in str(type(model)).lower():  # XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample_float)
            
        else:
            # Fallback to Kernel explainer
            explainer = shap.KernelExplainer(model.predict_proba, X_sample_float)
            shap_values = explainer.shap_values(X_sample_float)
        
        # Handle binary classification output
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Take positive class
        
        # Validation
        assert np.any(shap_values != 0), "SHAP values all zero"
        assert np.abs(shap_values).max() > 0.001, "SHAP magnitudes too small"
        
        logger.info(f"✅ SHAP computation complete. Shape: {shap_values.shape}")
        
        return explainer, shap_values, X_sample
        
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        raise ValueError(f"SHAP computation failed: {e}")


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: str
) -> None:
    """Plot and save SHAP summary."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=X_sample.columns.tolist(),
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ SHAP summary saved: {output_path}")


def plot_shap_bar(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: str
) -> None:
    """Plot and save SHAP bar chart."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=X_sample.columns.tolist(),
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ SHAP bar chart saved: {output_path}")


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    index: int = 0,
    output_path: Optional[str] = None
) -> None:
    """Plot waterfall plot for a single prediction."""
    plt.figure(figsize=(12, 6))
    
    # Create explanation object
    exp = shap.Explanation(
        values=shap_values[index],
        base_values=0,
        data=X_sample.iloc[index].values,
        feature_names=X_sample.columns.tolist()
    )
    
    shap.waterfall_plot(exp, show=False)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Waterfall plot generated")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("SHAP ANALYSIS FOR CREDIT RISK MODELS")
    print("="*60)
    
    # Choose model: 'xgboost', 'random_forest', 'gradient_boosting', or 'logistic'
    model_name = 'xgboost'  # Change this to analyze different models
    
    print(f"\n📊 Analyzing {model_name.upper()} model...")
    
    try:
        # Load data (automatically drops ID columns)
        X_test, y_test, feature_names = load_test_data()
        
        # Load model
        model = load_model(model_name)
        
        # Compute SHAP
        explainer, shap_values, X_sample = compute_shap_values(model, X_test)
        
        # Generate plots
        plot_shap_summary(
            shap_values, 
            X_sample,
            os.path.join(REPORTS_DIR, f"shap_summary_{model_name}.png")
        )
        plot_shap_bar(
            shap_values,
            X_sample,
            os.path.join(REPORTS_DIR, f"shap_bar_{model_name}.png")
        )
        
        # Waterfall for first few predictions
        for i in range(min(3, len(X_sample))):
            plot_shap_waterfall(
                shap_values,
                X_sample,
                index=i,
                output_path=os.path.join(REPORTS_DIR, f"shap_waterfall_{model_name}_{i}.png")
            )
        
        # Save SHAP values
        shap_df = pd.DataFrame(
            shap_values,
            columns=X_sample.columns,
            index=X_sample.index
        )
        shap_df.to_csv(os.path.join(REPORTS_DIR, f"shap_values_{model_name}.csv"))
        
        print(f"\n✅ SHAP analysis complete!")
        print(f"   Files saved in: {REPORTS_DIR}")
        print(f"   - Summary plot: shap_summary_{model_name}.png")
        print(f"   - Bar chart: shap_bar_{model_name}.png")
        print(f"   - SHAP values: shap_values_{model_name}.csv")
        
        # Show top features
        top_features = pd.DataFrame({
            'feature': X_sample.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False).head(10)
        
        print(f"\n📈 Top 10 Features by SHAP importance:")
        print(top_features.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()