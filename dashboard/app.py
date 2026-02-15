import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                           confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay)
import shap
import numpy as np
from src.data.preprocess import preprocess_and_split
from src.models.train import train_models
from src.inference.predict import predict
from src.explainability.shap_analysis import compute_shap_values

# Page config
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk Assessment Dashboard")
st.markdown("---")

@st.cache_resource
def load_data_and_model():
    """Load data and train model (cached for performance)."""
    with st.spinner("Loading data and training models..."):
        X_train, y_train, X_test, y_test, feature_names, pipeline = preprocess_and_split()
        results, _ = train_models()
        best_model = joblib.load("models/best_model.joblib")
    return X_test, y_test, best_model, pipeline, feature_names

# Load data
X_test, y_test, model, pipeline, feature_names = load_data_and_model()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Model Performance", "🎯 Predict", "📈 Explainability"])

if page == "📊 Model Performance":
    st.header("Model Performance Metrics")
    
    # Calculate metrics
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{auc:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1-Score", f"{f1:.3f}")
    
    st.markdown("---")
    
    # ROC Curve and Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_title(f'ROC Curve (AUC = {auc:.3f})')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, 
            display_labels=['Low Risk', 'High Risk'],
            ax=ax,
            cmap='Blues'
        )
        st.pyplot(fig)

elif page == "🎯 Predict":
    st.header("Risk Prediction Demo")
    st.markdown("Enter transaction details to get risk assessment:")
    
    # Input form with YOUR columns
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            customer_id = st.number_input("Customer ID", value=12345)
            amount = st.number_input("Transaction Amount ($)", value=250.00)
            value = st.number_input("Transaction Value ($)", value=225.00)
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP"])
            country = st.selectbox("Country", ["US", "UK", "DE"])
            
        with col2:
            provider = st.selectbox("Provider", ["Prov1", "Prov2", "Prov3"])
            category = st.selectbox("Product Category", 
                                   ["Electronics", "Clothing", "Food", "Travel"])
            channel = st.selectbox("Channel", ["Online", "Store", "Mobile"])
            pricing = st.selectbox("Pricing Strategy", ["Fixed", "Dynamic"])
            trans_time = st.text_input("Transaction Time", "2024-01-15 14:30:00")
        
        submitted = st.form_submit_button("Predict Risk")
    
    if submitted:
        # Prepare input data
        input_data = [{
            "CustomerId": customer_id,
            "TransactionStartTime": trans_time,
            "Amount": amount,
            "Value": value,
            "CurrencyCode": currency,
            "CountryCode": country,
            "ProviderId": provider,
            "ProductCategory": category,
            "ChannelId": channel,
            "PricingStrategy": pricing
        }]
        
        # Make prediction
        try:
            prob = predict(model, pipeline, input_data)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric("Default Probability", f"{prob:.1%}")
                
                if prob > 0.5:
                    st.error("⚠️ HIGH RISK")
                    st.markdown("""
                    **Recommendation:** 
                    - Request additional collateral
                    - Consider higher interest rate
                    - Flag for manual review
                    """)
                else:
                    st.success("✅ LOW RISK")
                    st.markdown("""
                    **Recommendation:** 
                    - Proceed with standard terms
                    - Monitor transaction patterns
                    """)
                    
        except Exception as e:
            st.error(f"Error making prediction: {e}")

else:  # Explainability
    st.header("Model Explainability with SHAP")
    st.markdown("Understanding what drives the predictions:")
    
    if st.button("Generate SHAP Analysis"):
        with st.spinner("Computing SHAP values..."):
            try:
                from src.explainability.shap_analysis import compute_shap_values
                
                # Compute SHAP values
                explainer, shap_values = compute_shap_values(
                    model, X_test, feature_names, sample_size=100
                )
                
                # Summary plot
                st.subheader("Feature Importance Summary")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, 
                    X_test.iloc[:100], 
                    feature_names=feature_names,
                    show=False
                )
                st.pyplot(fig)
                
                # Bar plot
                st.subheader("Mean Absolute SHAP Values")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, 
                    X_test.iloc[:100], 
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False
                )
                st.pyplot(fig)
                
                # Interpretation
                st.markdown("---")
                st.subheader("Interpretation")
                st.markdown("""
                - **SHAP values** show how each feature contributes to the prediction
                - **Red** = higher feature value, **Blue** = lower feature value
                - Features at the top are most important for the model
                - Positive SHAP = increases risk score, Negative = decreases risk
                """)
                
            except Exception as e:
                st.error(f"Error generating SHAP analysis: {e}")