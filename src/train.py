# src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os
from datetime import datetime

# Custom modules
from src.data_processing import create_data_pipeline
from src.target_engineering import create_proxy_target

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)

    # Generate target
    target_df = create_proxy_target(df, snapshot_date=None)

    # Ensure CustomerId exists
    if "CustomerId" not in df.columns:
        df["CustomerId"] = df.index

    # Convert CustomerId to integer safely
    df["CustomerId"] = df["CustomerId"].astype(str).str.extract(r'(\d+)').astype(int)
    target_df["CustomerId"] = target_df["CustomerId"].astype(str).str.extract(r'(\d+)').astype(int)

    # Merge and drop missing target
    df = df.merge(target_df, on="CustomerId", how="left")
    df = df.dropna(subset=["is_high_risk"])

    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    return X, y


def train_models(X, y):
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment("credit_risk_modeling")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = create_data_pipeline()
    X_train_proc = pipeline.fit_transform(X_train, y_train)
    X_test_proc = pipeline.transform(X_test)

    models = {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01,0.1,1,10], "penalty": ["l1","l2"], "solver":["liblinear"]}
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators":[100,200], "max_depth":[10,20,None], "min_samples_split":[2,5,10]}
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators":[100,200], "learning_rate":[0.01,0.1,0.2], "max_depth":[3,5,7]}
        }
    }

    best_models = {}

    for name, cfg in models.items():
        with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print(f"\nTraining {name}...")
            gs = GridSearchCV(cfg["model"], cfg["params"], cv=5, scoring="roc_auc", n_jobs=-1, verbose=1)
            gs.fit(X_train_proc, y_train)

            best = gs.best_estimator_
            y_pred = best.predict(X_test_proc)
            y_proba = best.predict_proba(X_test_proc)[:,1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }

            mlflow.log_params(gs.best_params_)
            for k,v in metrics.items():
                mlflow.log_metric(k,v)
            mlflow.sklearn.log_model(best, name)
            mlflow.log_dict(classification_report(y_test,y_pred,output_dict=True), f"{name}_report.json")

            best_models[name] = {"model": best, "metrics": metrics, "params": gs.best_params_}
            print(f"Best params: {gs.best_params_}")
            print(f"Metrics: {metrics}")

    return best_models, pipeline

def save_best_model(best_models, pipeline):
    # Select best model by ROC-AUC
    best_name = max(best_models, key=lambda m: best_models[m]["metrics"]["roc_auc"])
    best_model = best_models[best_name]["model"]

    joblib.dump(best_model, f"models/{best_name}_model.pkl")
    joblib.dump(pipeline, "models/data_pipeline.pkl")
    print(f"\nSaved best model '{best_name}' and pipeline to 'models/' folder.")
    return best_name

def main():
    data_path = "data/raw/data.csv"
    print("Loading data...")
    X, y = load_and_prepare_data(data_path)

    print("Training models...")
    best_models, pipeline = train_models(X, y)

    print("Saving best model and pipeline...")
    save_best_model(best_models, pipeline)

    print("Training completed!")

if __name__ == "__main__":
    main()
