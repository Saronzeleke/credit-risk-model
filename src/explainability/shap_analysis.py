import shap
import matplotlib.pyplot as plt
import joblib
from typing import Any, List, Tuple
import pandas as pd
import numpy as np


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    sample_size: int = 100
) -> Tuple[Any, np.ndarray]:
    """
    Compute SHAP values with validation.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        sample_size: Number of samples to use
    
    Returns:
        explainer: SHAP explainer
        shap_values: Computed SHAP values
    """
    # Sample data for efficiency
    X_sample = X.sample(min(sample_size, len(X)), random_state=42)
    
    try:
        # Choose appropriate explainer based on model type
        if hasattr(model, 'coef_'):  # Linear models
            explainer = shap.LinearExplainer(model, X_sample)
        elif hasattr(model, 'estimators_'):  # Tree-based
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback to Kernel explainer
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification output
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Take positive class
        
        # Validation
        assert np.any(shap_values != 0), "SHAP values all zero"
        assert np.abs(shap_values).max() > 0.001, "SHAP magnitudes too small"
        
        return explainer, shap_values
        
    except Exception as e:
        raise ValueError(f"SHAP computation failed: {e}")


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    output_path: str = 'reports/shap_summary.png'
) -> None:
    """
    Plot and save SHAP summary.
    """
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_shap_bar(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    output_path: str = 'reports/shap_bar.png'
) -> None:
    """
    Plot and save SHAP bar chart.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test SHAP computation
    from src.data.preprocess import preprocess_and_split
    from src.models.train import train_models
    
    print("Loading data and model...")
    _, _, X_test, y_test, feature_names, pipeline = preprocess_and_split()
    pipeline = joblib.load(r"C:\Users\admin\credit-risk-model\models\data_pipeline.pkl")
    model = joblib.load(r"C:\Users\admin\credit-risk-model\models\gradient_boosting_model.pkl")

    
    print("Computing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X_test, feature_names)
    
    print("Generating plots...")
    plot_shap_summary(shap_values, X_test.iloc[:100], feature_names)
    plot_shap_bar(shap_values, X_test.iloc[:100], feature_names)
    
    print("SHAP analysis complete! Check reports/ directory.")