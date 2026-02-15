import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, RocCurveDisplay, ConfusionMatrixDisplay
import shap
import io

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk Assessment Dashboard")
st.markdown("---")

# -------------------------------
# Load model and data
# -------------------------------
@st.cache_resource
def load_model_and_data(model_name="Logistic Regression"):
    """Load trained model, pipeline, and test data (cached)."""
    try:
        # Choose model
        if model_name == "Logistic Regression":
            model = joblib.load("models/logistic_model.pkl")
        elif model_name == "Random Forest":
            model = joblib.load("models/random_forest_model.pkl")
        else:  # Gradient Boosting
            model = joblib.load("models/gradient_boosting_model.pkl")
        
        # Load preprocessing pipeline
        pipeline = joblib.load("models/data_pipeline.pkl")
        
        # Load test data
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv")["is_high_risk"]
        feature_names = X_test.columns.tolist()
        
        return X_test, y_test, model, pipeline, feature_names
    except Exception as e:
        st.error(f"Error loading model/data: {e}")
        return None, None, None, None, None

# -------------------------------
# Sidebar navigation & model selection
# -------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Model Performance", "🎯 Predict", "📈 Explainability"])

# Model selection
model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest", "Gradient Boosting"]
)

X_test, y_test, model, pipeline, feature_names = load_model_and_data(model_option)

# Stop execution if model failed to load
if model is None:
    st.stop()

# -------------------------------
# Model Performance Page
# -------------------------------
if page == "📊 Model Performance":
    st.header(f"Model Performance Metrics ({model_option})")
    
    y_pred_proba = model.predict_proba(pipeline.transform(X_test))[:, 1]
    y_pred = model.predict(pipeline.transform(X_test))
    
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{auc:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1-Score", f"{f1:.3f}")
    
    st.markdown("---")
    
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
            y_test, y_pred, display_labels=['Low Risk', 'High Risk'], ax=ax, cmap='Blues'
        )
        st.pyplot(fig)

# -------------------------------
# Predict Page
# -------------------------------
elif page == "🎯 Predict":
    st.header("Risk Prediction Demo")
    st.markdown("Enter transaction details to get risk assessment:")
    
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
            category = st.selectbox("Product Category", ["Electronics", "Clothing", "Food", "Travel"])
            channel = st.selectbox("Channel", ["Online", "Store", "Mobile"])
            pricing = st.selectbox("Pricing Strategy", ["Fixed", "Dynamic"])
            trans_time = st.text_input("Transaction Time", "2024-01-15 14:30:00")
        
        submitted = st.form_submit_button("Predict Risk")
    
    if submitted:
        input_data = pd.DataFrame([{
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
        }])
        
        try:
            input_transformed = pipeline.transform(input_data)
            prob = model.predict_proba(input_transformed)[:, 1][0]
            
            st.markdown("---")
            st.subheader("Prediction Result")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric("Default Probability", f"{prob:.1%}")
                if prob > 0.5:
                    st.error("⚠️ HIGH RISK")
                    st.markdown("""**Recommendation:** 
- Request additional collateral
- Consider higher interest rate
- Flag for manual review
""")
                else:
                    st.success("✅ LOW RISK")
                    st.markdown("""**Recommendation:** 
- Proceed with standard terms
- Monitor transaction patterns
""")
            
            # Download prediction CSV
            csv_buffer = io.StringIO()
            input_data["Default_Probability"] = prob
            input_data.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Prediction CSV",
                data=csv_buffer.getvalue(),
                file_name=f"prediction_{customer_id}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# -------------------------------
# Explainability Page
# -------------------------------
else:
    st.header(f"Model Explainability with SHAP ({model_option})")
    st.markdown("Understanding what drives the predictions:")

    @st.cache_resource
    def get_shap_values(model, X, sample_size=100):
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X.sample(sample_size, random_state=42))
        return shap_values

    if st.button("Generate SHAP Analysis"):
        with st.spinner("Computing SHAP values..."):
            try:
                shap_values = get_shap_values(model, X_test)
                
                st.subheader("Feature Importance Summary")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
                st.pyplot(fig)
                
                st.subheader("Mean Absolute SHAP Values")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test.iloc[:100], plot_type="bar", show=False)
                st.pyplot(fig)
                
                st.markdown("---")
                st.subheader("Interpretation")
                st.markdown("""
- **SHAP values** show feature contribution per prediction
- Red = higher feature value, Blue = lower
- Top features impact model most
- Positive SHAP = increases risk, Negative = decreases
""")
                
                # Download SHAP CSV
                shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
                shap_df["prediction"] = shap_values.base_values + shap_values.values.sum(axis=1)
                csv_buffer = io.StringIO()
                shap_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download SHAP Values CSV",
                    data=csv_buffer.getvalue(),
                    file_name="shap_values.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error generating SHAP analysis: {e}")
