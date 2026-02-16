import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import logging
import sys
import os
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionPredictor:
    """Production-ready predictor for credit risk/fraud detection."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_info = {}
        self.feature_names = None
        self.id_columns = ['TransactionId', 'BatchId', 'AccountId',
                           'SubscriptionId', 'CustomerId', 'ProductId']
        self.target_column = 'FraudResult'

        if model_path:
            self.load_model(model_path)
        else:
            # Auto-detect best model
            self.load_best_model()
            
        logger.info("FraudDetectionPredictor initialized")

    def load_model(self, model_path: str):
        """Load trained model from joblib file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)
        
        # Handle different model save formats
        if isinstance(model_data, dict):
            self.model = model_data.get('model', model_data)
            self.model_info = {
                'model_type': model_data.get('model_type', Path(model_path).stem.replace('_model', '')),
                'best_params': model_data.get('best_params', {}),
                'metrics': model_data.get('metrics', {}),
                'threshold': model_data.get('threshold', 0.5),
                'feature_names': model_data.get('feature_names')
            }
        else:
            self.model = model_data
            self.model_info = {
                'model_type': Path(model_path).stem.replace('_model', ''),
                'threshold': 0.5
            }

        # Get feature names from model
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        elif self.model_info.get('feature_names'):
            self.feature_names = self.model_info['feature_names']
        else:
            # Fallback - will be set during first prediction
            self.feature_names = None

        logger.info(f"✅ Model loaded: {model_path}")
        logger.info(f"   Type: {self.model_info.get('model_type', 'unknown')}")
        logger.info(f"   Threshold: {self.model_info.get('threshold', 0.5):.3f}")

    def load_best_model(self, models_dir: str = r'C:\Users\admin\credit-risk-model\models'):
        """Automatically load the best performing model."""
        model_files = list(Path(models_dir).glob('*_model.pkl'))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        
        # Prefer models in order: xgboost > random_forest > gradient_boosting > logistic
        preferred_order = ['xgboost', 'random_forest', 'gradient_boosting', 'logistic']
        
        for model_name in preferred_order:
            model_path = Path(models_dir) / f"{model_name}_model.pkl"
            if model_path.exists():
                logger.info(f"📊 Loading best available model: {model_name}")
                self.load_model(str(model_path))
                return
        
        # Fallback to first available
        self.load_model(str(model_files[0]))

    def _get_required_features(self) -> list:
        """Get the list of features the model expects."""
        if self.feature_names is not None:
            return self.feature_names
        
        # Try to infer from model
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        elif hasattr(self.model, 'n_features_in_'):
            # Can't get names, will need to rely on order
            pass
        
        return self.feature_names or []

    def _clean_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove ID columns and keep only numeric features."""
        df_clean = df.copy()
        
        # Drop ID columns if they exist
        cols_to_drop = [col for col in self.id_columns + [self.target_column] 
                       if col in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            logger.debug(f"Dropped columns: {cols_to_drop}")
        
        # Keep only numeric columns for prediction
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df_clean[numeric_cols]
        
        logger.debug(f"Using numeric features: {numeric_cols}")
        return df_numeric

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align input features to match training feature names."""
        required_features = self._get_required_features()
        
        if not required_features:
            # If we don't know feature names, return as-is
            return df
        
        df_aligned = pd.DataFrame(index=df.index)
        
        # Add required features (fill missing with 0)
        for feature in required_features:
            if feature in df.columns:
                df_aligned[feature] = df[feature]
            else:
                logger.debug(f"Missing feature '{feature}' - filling with 0")
                df_aligned[feature] = 0
        
        logger.debug(f"Aligned features: {df_aligned.shape[1]} features")
        return df_aligned

    def predict(self, df: pd.DataFrame, return_proba: bool = False):
        """Generate predictions."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Clean and prepare data
        df_clean = self._clean_input(df)
        df_aligned = self._align_features(df_clean)
        
        # Convert to float32 for XGBoost compatibility
        df_aligned = df_aligned.astype(np.float32)
        
        # Predict
        if return_proba:
            proba = self.model.predict_proba(df_aligned)[:, 1]
            return proba
        else:
            threshold = self.model_info.get('threshold', 0.5)
            proba = self.model.predict_proba(df_aligned)[:, 1]
            return (proba >= threshold).astype(int)

    def predict_with_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions with additional metadata."""
        # Keep IDs for reference
        ids_df = df[[col for col in self.id_columns if col in df.columns]].copy()
        
        # Get predictions
        probabilities = self.predict(df, return_proba=True)
        threshold = self.model_info.get('threshold', 0.5)
        predictions = (probabilities >= threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'probability': probabilities,
            'prediction': predictions,
            'risk_level': pd.cut(probabilities, 
                                 bins=[0, 0.3, 0.7, 1.0], 
                                 labels=['Low', 'Medium', 'High'])
        })
        
        # Add IDs if available
        if not ids_df.empty:
            results = pd.concat([ids_df.reset_index(drop=True), 
                                 results.reset_index(drop=True)], axis=1)
        
        # Add model info
        results.attrs['model_type'] = self.model_info.get('model_type', 'unknown')
        results.attrs['threshold'] = threshold
        
        return results

    def predict_batch(self, input_csv: str, output_csv: Optional[str] = None) -> pd.DataFrame:
        """Process batch predictions from CSV file."""
        logger.info(f"📥 Reading input file: {input_csv}")
        df = pd.read_csv(input_csv)
        
        logger.info(f"📊 Processing {len(df)} transactions...")
        results = self.predict_with_metadata(df)
        
        if output_csv:
            # Create output directory if needed
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_csv, index=False)
            logger.info(f"✅ Predictions saved to: {output_csv}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fraud/Credit Risk Prediction")
    parser.add_argument('--input', required=True, help='Input CSV file with transactions')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV file for predictions')
    parser.add_argument('--model', help='Specific model file to use (optional)')
    parser.add_argument('--threshold', type=float, help='Custom probability threshold')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize predictor
        if args.model:
            predictor = FraudDetectionPredictor(model_path=args.model)
        else:
            predictor = FraudDetectionPredictor()
        
        # Override threshold if provided
        if args.threshold is not None:
            predictor.model_info['threshold'] = args.threshold
            logger.info(f"⚙️ Using custom threshold: {args.threshold}")
        
        # Run predictions
        results = predictor.predict_batch(args.input, args.output)
        
        # Show summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Model: {predictor.model_info.get('model_type', 'unknown')}")
        print(f"Threshold: {predictor.model_info.get('threshold', 0.5):.3f}")
        print(f"\nRisk Distribution:")
        print(results['risk_level'].value_counts())
        print(f"\nSample predictions:")
        print(results.head(10))
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()