import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                            f1_score, RocCurveDisplay, ConfusionMatrixDisplay,
                            classification_report)
import shap
import io
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Page config

st.set_page_config(
    page_title="Credit Risk Assessment Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .risk-high {
        color: #DC2626;
        font-weight: 600;
    }
    .risk-low {
        color: #059669;
        font-weight: 600;
    }
    .risk-medium {
        color: #D97706;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODELS_DIR = Path(r"C:/Users/admin/credit-risk-model/models")
PROCESSED_DATA_DIR = Path(r"C:/Users/admin/credit-risk-model/data/processed")
REPORTS_DIR = Path(r"C:/Users/admin/credit-risk-model/reports")

# Create directories if they don't exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Model mapping
MODEL_MAP = {
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

# Load model and data
@st.cache_resource
def load_model_and_data(model_name="XGBoost"):
    """Load trained model, and test data (cached)."""
    try:
        # Get model file
        model_file = MODEL_MAP[model_name]
        model_path = MODELS_DIR / model_file
        
        if not model_path.exists():
            st.error(f"Model not found: {model_path}")
            return None, None, None, None
        
        # Load model
        model_data = joblib.load(model_path)
        
        # Handle different save formats
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            model_info = {
                'metrics': model_data.get('metrics', {}),
                'threshold': model_data.get('threshold', 0.5)
            }
        else:
            model = model_data
            model_info = {'threshold': 0.5}
        
        # Load test data
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
        y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()
        
        # Drop ID columns if present
        id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
                   'CustomerId', 'ProductId']
        X_test = X_test.drop(columns=[c for c in id_cols if c in X_test.columns], 
                            errors='ignore')
        
        # Keep only numeric
        X_test = X_test.select_dtypes(include=[np.number])
        
        feature_names = X_test.columns.tolist()
        
        st.sidebar.success(f"✅ Loaded {model_name}")
        st.sidebar.info(f"📊 Test data: {X_test.shape}")
        
        return X_test, y_test, model, feature_names, model_info
        
    except Exception as e:
        st.error(f"Error loading model/data: {e}")
        return None, None, None, None, None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
    st.title("🏦 Credit Risk Dashboard")
    st.markdown("---")
    
    # Model selection
    model_option = st.selectbox(
        "🔍 Select Model",
        list(MODEL_MAP.keys()),
        index=3  # Default to XGBoost
    )
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📌 Navigation",
        ["📊 Model Performance", "🎯 Predict", "📈 Explainability", "📋 Batch Prediction"]
    )
    
    st.markdown("---")
    
    # Threshold adjustment
    threshold = st.slider(
        "⚙️ Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    st.markdown("---")
    st.caption("© 2026 Credit Risk Analytics")

# Load data
X_test, y_test, model, feature_names, model_info = load_model_and_data(model_option)

if model is None:
    st.stop()
# Helper functions
def get_predictions(X):
    """Get predictions and probabilities."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred

def get_risk_level(prob):
    """Categorize risk level."""
    if prob < 0.3:
        return "Low", "#059669"
    elif prob < 0.7:
        return "Medium", "#D97706"
    else:
        return "High", "#DC2626"

# Model Performance Page
if page == "📊 Model Performance":
    st.markdown("<h1 class='main-header'>📊 Model Performance</h1>", unsafe_allow_html=True)
    
    # Get predictions
    y_pred_proba, y_pred = get_predictions(X_test)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("AUC-ROC", f"{auc:.3f}", 
                  delta="0.873" if auc > 0.85 else None)
    
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    
    with col5:
        st.metric("Threshold", f"{threshold:.2f}")
    
    st.markdown("---")
    
    # Classification Report
    with st.expander("📋 Detailed Classification Report", expanded=False):
        report = classification_report(y_test, y_pred, 
                                      target_names=['Low Risk', 'High Risk'],
                                      output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_title(f'ROC Curve (AUC = {auc:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Download ROC data
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        csv = roc_df.to_csv(index=False)
        st.download_button("📥 Download ROC Data", csv, "roc_data.csv", "text/csv")
    
    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, 
            display_labels=['Low Risk', 'High Risk'],
            ax=ax, 
            cmap='Blues',
            colorbar=False
        )
        st.pyplot(fig)
        
        # Download predictions
        pred_df = pd.DataFrame({
            'true_label': y_test,
            'predicted': y_pred,
            'probability': y_pred_proba
        })
        csv = pred_df.head(1000).to_csv(index=False)
        st.download_button("📥 Download Sample Predictions", csv, 
                          "sample_predictions.csv", "text/csv")

# Predict Page
elif page == "🎯 Predict":
    st.markdown("<h1 class='main-header'>🎯 Single Transaction Prediction</h1>", 
                unsafe_allow_html=True)
    
    st.info("Enter transaction details below to get instant risk assessment")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 Transaction Details")
            amount = st.number_input("Amount ($)", value=250.00, min_value=0.0)
            value = st.number_input("Value ($)", value=225.00, min_value=0.0)
            trans_hour = st.slider("Transaction Hour", 0, 23, 14)
            trans_day = st.slider("Day of Month", 1, 31, 15)
            trans_month = st.slider("Month", 1, 12, 1)
            day_of_week = st.selectbox("Day of Week", 
                                       ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                        "Friday", "Saturday", "Sunday"])
        
        with col2:
            st.subheader("🏷️ Product Information")
            provider = st.selectbox("Provider", ["Provider_1", "Provider_2", "Provider_3", 
                                                 "Provider_4", "Provider_5", "Provider_6"])
            category = st.selectbox("Product Category", 
                                   ["airtime", "financial_services", "utility_bill", 
                                    "data_bundles", "tv", "movies", "transport", "ticket"])
            pricing = st.selectbox("Pricing Strategy", [1, 2, 3, 4])
            channel = st.selectbox("Channel", ["Channel_1", "Channel_2", "Channel_3", 
                                               "Channel_4", "Channel_5"])
        
        with col3:
            st.subheader("🌍 Location")
            country = st.selectbox("Country Code", [256, 254, 255, 257])
            currency = st.selectbox("Currency", ["UGX", "KES", "TZS", "USD"])
            customer_id = st.number_input("Customer ID", value=12345)
            transaction_id = st.text_input("Transaction ID", "TXN001")
        
        submitted = st.form_submit_button("🚀 Predict Risk", use_container_width=True)
    
    if submitted:
        # Map day of week to numeric
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'Amount': amount,
            'Value': value,
            'CountryCode': country,
            'PricingStrategy': pricing,
            'TransactionHour': trans_hour,
            'TransactionDay': trans_day,
            'TransactionMonth': trans_month,
            'TransactionDayOfWeek': day_map[day_of_week]
        }])
        
        try:
            # Predict
            prob = model.predict_proba(input_data)[0, 1]
            risk_level, color = get_risk_level(prob)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Probability</h3>
                    <h2>{prob:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Risk Level</h3>
                    <h2 style='color: {color}'>{risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                decision = "❌ REJECT" if prob > threshold else "✅ APPROVE"
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Decision</h3>
                    <h2>{decision}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if prob > 0.7:
                st.error("""
                **High Risk - Action Required:**
                - 🚫 Decline application
                - 📋 Flag for fraud review
                - 🔒 Freeze account if suspicious
                """)
            elif prob > 0.3:
                st.warning("""
                **Medium Risk - Additional Verification:**
                - 📞 Verify identity
                - 📧 Send confirmation email
                - 📊 Monitor transaction pattern
                """)
            else:
                st.success("""
                **Low Risk - Proceed:**
                - ✅ Approve transaction
                - 📈 Update customer profile
                - 💳 Standard processing
                """)
            
            # Download result
            result_df = input_data.copy()
            result_df['probability'] = prob
            result_df['risk_level'] = risk_level
            result_df['decision'] = decision
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                "📥 Download Prediction Report",
                csv,
                f"prediction_{transaction_id}.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
# Explainability Page
elif page == "📈 Explainability":
    st.markdown("<h1 class='main-header'>📈 Model Explainability</h1>", 
                unsafe_allow_html=True)
    
    st.info("Understanding what drives the model's predictions using SHAP")
    
    # Sample size selector
    sample_size = st.slider("Sample Size", 50, 500, 100, 50)
    
    if st.button("🔍 Generate SHAP Analysis", use_container_width=True):
        with st.spinner("Computing SHAP values (this may take a minute)..."):
            try:
                # Sample data
                X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
                
                # Create explainer
                if 'xgb' in str(model).lower():
                    explainer = shap.TreeExplainer(model)
                elif 'randomforest' in str(model).lower() or 'gradientboosting' in str(model).lower():
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.LinearExplainer(model, X_sample)
                
                # Compute SHAP values
                shap_values = explainer.shap_values(X_sample.astype(np.float32))
                
                # Handle binary output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Summary plot
                st.subheader("Feature Impact Summary")
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Bar plot
                st.subheader("Feature Importance (Mean |SHAP|)")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Top features table
                st.subheader("Top 10 Most Important Features")
                importance_df = pd.DataFrame({
                    'Feature': X_sample.columns,
                    'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
                }).sort_values('Mean |SHAP|', ascending=False).head(10)
                
                st.dataframe(importance_df)
                
                # Waterfall for single prediction
                st.subheader("Single Prediction Breakdown")
                idx = st.number_input("Select sample index", 0, len(X_sample)-1, 0)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[idx],
                        base_values=0,
                        data=X_sample.iloc[idx].values,
                        feature_names=X_sample.columns.tolist()
                    ),
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download SHAP values
                shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
                shap_df['probability'] = model.predict_proba(X_sample)[:, 1]
                
                csv = shap_df.to_csv(index=False)
                st.download_button(
                    "📥 Download SHAP Values",
                    csv,
                    "shap_values.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")
# Batch Prediction Page
else:
    st.markdown("<h1 class='main-header'>📋 Batch Prediction</h1>", 
                unsafe_allow_html=True)
    
    st.info("Upload a CSV file with multiple transactions for batch processing")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read file
            df = pd.read_csv(uploaded_file)
            st.write(f"📊 File contains {len(df)} transactions")
            st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner("Processing..."):
                    # Prepare data
                    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
                              'CustomerId', 'ProductId']
                    ids_df = df[[c for c in id_cols if c in df.columns]].copy()
                    
                    # Select numeric features
                    numeric_cols = ['Amount', 'Value', 'CountryCode', 'PricingStrategy',
                                   'TransactionHour', 'TransactionDay', 'TransactionMonth',
                                   'TransactionDayOfWeek']
                    
                    X_pred = df[[c for c in numeric_cols if c in df.columns]]
                    
                    # Predict
                    proba = model.predict_proba(X_pred)[:, 1]
                    pred = (proba >= threshold).astype(int)
                    
                    # Create results
                    results = pd.DataFrame({
                        'probability': proba,
                        'prediction': pred,
                        'risk_level': ['Low' if p < 0.3 else 'Medium' if p < 0.7 else 'High' 
                                      for p in proba]
                    })
                    
                    # Add IDs
                    if not ids_df.empty:
                        results = pd.concat([ids_df.reset_index(drop=True), 
                                            results.reset_index(drop=True)], axis=1)
                    
                    # Summary
                    st.success("✅ Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("High Risk", (results['risk_level'] == 'High').sum())
                    col2.metric("Medium Risk", (results['risk_level'] == 'Medium').sum())
                    col3.metric("Low Risk", (results['risk_level'] == 'Low').sum())
                    
                    # Show results
                    st.subheader("Predictions")
                    st.dataframe(results)
                    
                    # Download
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions",
                        csv,
                        "batch_predictions.csv",
                        "text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")