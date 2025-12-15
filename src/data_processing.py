# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Temporal Feature Engineering
# --------------------------------------------------
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column="TransactionStartTime"):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column], errors="coerce")

        X["transaction_hour"] = X[self.date_column].dt.hour
        X["transaction_day"] = X[self.date_column].dt.day
        X["transaction_month"] = X[self.date_column].dt.month
        X["transaction_year"] = X[self.date_column].dt.year
        X["transaction_dayofweek"] = X[self.date_column].dt.dayofweek
        X["is_weekend"] = X["transaction_dayofweek"].isin([5, 6]).astype(int)

        return X.drop(columns=[self.date_column])


# --------------------------------------------------
# Aggregate Feature Engineering
# --------------------------------------------------
class AggregateFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_column="CustomerId"):
        self.groupby_column = groupby_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        agg = X.groupby(self.groupby_column).agg(
            Amount_sum=("Amount", "sum"),
            Amount_mean=("Amount", "mean"),
            Amount_std=("Amount", "std"),
            tx_count=("Amount", "count"),
            Value_sum=("Value", "sum"),
            Value_mean=("Value", "mean"),
            Value_std=("Value", "std"),
        ).fillna(0)

        agg["amount_value_ratio"] = agg["Amount_sum"] / (agg["Value_sum"] + 1e-6)
        agg["transaction_frequency"] = agg["tx_count"] / 30.0

        return X.merge(agg, on=self.groupby_column, how="left")


# --------------------------------------------------
# WOE Transformer (SUPERVISED & SAFE)
# --------------------------------------------------
class WoeTransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence transformer.
    - Fits ONLY when y is provided
    - Safe for inference (no y)
    """

    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.woe_maps_ = {}

    def fit(self, X, y=None):
        if y is None:
            # Inference mode — do nothing
            return self

        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        for col in self.categorical_features:
            stats = (
                pd.concat([X[col], y], axis=1)
                .groupby(col)[y.name]
                .agg(["count", "sum"])
                .rename(columns={"sum": "bad"})
            )

            stats["good"] = stats["count"] - stats["bad"]
            stats["good_rate"] = stats["good"] / (total_good + 1e-6)
            stats["bad_rate"] = stats["bad"] / (total_bad + 1e-6)

            stats["woe"] = np.log(
                (stats["good_rate"] + 1e-6) /
                (stats["bad_rate"] + 1e-6)
            )

            self.woe_maps_[col] = stats["woe"].to_dict()

        return self

    def transform(self, X):
        X = X.copy()

        for col, mapping in self.woe_maps_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(0)

        return X


# --------------------------------------------------
# Pipeline Factory
# --------------------------------------------------
def create_data_pipeline():

    categorical_features = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]

    numerical_features = [
        "Amount",
        "Value",
        "Amount_sum",
        "Amount_mean",
        "Amount_std",
        "tx_count",
        "Value_sum",
        "Value_mean",
        "Value_std",
        "amount_value_ratio",
        "transaction_frequency",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
        "transaction_dayofweek",
        "is_weekend",
    ]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("temporal", TemporalFeatureExtractor()),
            ("aggregate", AggregateFeatureEngineer()),
            ("woe", WoeTransformer(categorical_features)),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


# --------------------------------------------------
# Public API
# --------------------------------------------------
def process_data(
    raw_data_path,
    target_column=None,
    pipeline=None,
    fit=True,
):
    df = pd.read_csv(raw_data_path)

    if pipeline is None:
        pipeline = create_data_pipeline()

    if fit and target_column:
        y = df[target_column]
        X = df.drop(columns=[target_column])
        X_transformed = pipeline.fit_transform(X, y)
    elif fit:
        X_transformed = pipeline.fit_transform(df)
    else:
        X_transformed = pipeline.transform(df)

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    return X_transformed, feature_names, pipeline

