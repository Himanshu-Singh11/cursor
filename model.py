"""
Machine Learning model for exoplanet detection.
Uses Random Forest as baseline with feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ExoplanetDetector:
    """ML model for detecting exoplanets from light curve features."""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        
    def create_model(self, **kwargs) -> None:
        """Create the ML model."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',  # Handle class imbalance
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_columns: list) -> Dict[str, Any]:
        """Train the model."""
        if self.model is None:
            self.create_model()
        
        self.feature_columns = feature_columns
        
        print("Training exoplanet detection model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Get training metrics
        train_score = self.model.score(X_train, y_train)
        train_predictions = self.model.predict(X_train)
        train_probabilities = self.model.predict_proba(X_train)[:, 1]
        
        metrics = {
            'train_accuracy': train_score,
            'train_auc': roc_auc_score(y_train, train_probabilities),
            'feature_importance': dict(zip(feature_columns, self.model.feature_importances_))
        }
        
        print(f"Training completed!")
        print(f"Training Accuracy: {train_score:.3f}")
        print(f"Training AUC: {metrics['train_auc']:.3f}")
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_accuracy = self.model.score(X_test, y_test)
        test_auc = roc_auc_score(y_test, y_prob)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_prob.tolist()
        }
        
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Test AUC: {test_auc:.3f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def predict_single(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Predict on a single sample."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert features to array in correct order
        feature_array = np.array([features.get(col, 0.0) for col in self.feature_columns]).reshape(1, -1)
        
        prediction = self.model.predict(feature_array)[0]
        probability = self.model.predict_proba(feature_array)[0, 1]
        
        return int(prediction), float(probability)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")

def train_exoplanet_model(data_dir: str = "data", model_dir: str = "models") -> ExoplanetDetector:
    """Complete training pipeline for exoplanet detection model."""
    from data_processor import ExoplanetDataProcessor
    
    # Load and preprocess data
    processor = ExoplanetDataProcessor(data_dir)
    df = processor.load_data()
    X, y, feature_columns = processor.preprocess_data(df)
    X_train, X_test, y_train, y_test = processor.create_train_test_split(X, y)
    
    # Train model
    detector = ExoplanetDetector()
    detector.create_model()
    train_metrics = detector.train(X_train, y_train, feature_columns)
    
    # Evaluate model
    test_metrics = detector.evaluate(X_test, y_test)
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "exoplanet_detector.pkl")
    detector.save_model(model_path)
    
    print("\n" + "="*50)
    print("EXOPLANET DETECTION MODEL TRAINING COMPLETE")
    print("="*50)
    print(f"Training Accuracy: {train_metrics['train_accuracy']:.3f}")
    print(f"Test Accuracy: {test_metrics['test_accuracy']:.3f}")
    print(f"Test AUC: {test_metrics['test_auc']:.3f}")
    print(f"Model saved to: {model_path}")
    
    return detector

if __name__ == "__main__":
    # Train the model
    detector = train_exoplanet_model()
