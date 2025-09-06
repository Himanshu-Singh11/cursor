"""
Data preprocessing pipeline for NASA exoplanet datasets.
Handles Kepler and TESS light curve data processing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
import os
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class ExoplanetDataProcessor:
    """Processes NASA exoplanet datasets for ML training."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def download_sample_data(self) -> None:
        """Download sample Kepler data for demonstration."""
        # For demo purposes, we'll create synthetic data that mimics real Kepler data
        print("Creating synthetic exoplanet data for demonstration...")
        
        # Create synthetic light curve data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features that would be extracted from light curves
        data = {
            'period': np.random.exponential(10, n_samples),  # Orbital period in days
            'duration': np.random.normal(0.1, 0.05, n_samples),  # Transit duration
            'depth': np.random.exponential(0.01, n_samples),  # Transit depth
            'snr': np.random.normal(10, 5, n_samples),  # Signal-to-noise ratio
            'impact_param': np.random.uniform(0, 1, n_samples),  # Impact parameter
            'eccentricity': np.random.beta(2, 5, n_samples),  # Orbital eccentricity
            'stellar_radius': np.random.normal(1, 0.3, n_samples),  # Stellar radius
            'stellar_mass': np.random.normal(1, 0.2, n_samples),  # Stellar mass
            'stellar_teff': np.random.normal(5778, 1000, n_samples),  # Stellar temperature
            'is_planet': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Target variable
        }
        
        df = pd.DataFrame(data)
        
        # Add some realistic correlations
        df.loc[df['is_planet'] == 1, 'snr'] += np.random.normal(5, 2, df[df['is_planet'] == 1].shape[0])
        df.loc[df['is_planet'] == 1, 'depth'] *= np.random.uniform(1.5, 3, df[df['is_planet'] == 1].shape[0])
        
        # Save to CSV
        os.makedirs(self.data_dir, exist_ok=True)
        df.to_csv(f"{self.data_dir}/kepler_sample_data.csv", index=False)
        print(f"Sample data saved to {self.data_dir}/kepler_sample_data.csv")
        
    def load_data(self, filename: str = "kepler_sample_data.csv") -> pd.DataFrame:
        """Load exoplanet data from CSV file."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            self.download_sample_data()
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples with {df['is_planet'].sum()} confirmed planets")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess the data for ML training."""
        # Select features (exclude target variable)
        feature_columns = [col for col in df.columns if col != 'is_planet']
        self.feature_columns = feature_columns
        
        X = df[feature_columns].values
        y = df['is_planet'].values
        
        # Handle any missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Preprocessed data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y, feature_columns
    
    def create_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                              test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_data(self, feature_importances: np.ndarray) -> pd.DataFrame:
        """Create DataFrame for feature importance visualization."""
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

def create_sample_light_curve_data(n_points: int = 100) -> Dict:
    """Create sample light curve data for visualization."""
    # Simulate a transit light curve
    time = np.linspace(0, 2, n_points)
    
    # Create a transit signal
    transit_center = 1.0
    transit_width = 0.1
    transit_depth = 0.02
    
    flux = np.ones_like(time)
    transit_mask = (time >= transit_center - transit_width/2) & (time <= transit_center + transit_width/2)
    flux[transit_mask] = 1 - transit_depth
    
    # Add noise
    noise = np.random.normal(0, 0.005, len(time))
    flux += noise
    
    return {
        'time': time,
        'flux': flux,
        'transit_center': transit_center,
        'transit_width': transit_width,
        'transit_depth': transit_depth
    }

if __name__ == "__main__":
    # Test the data processor
    processor = ExoplanetDataProcessor()
    df = processor.load_data()
    X, y, features = processor.preprocess_data(df)
    X_train, X_test, y_train, y_test = processor.create_train_test_split(X, y)
    
    print("Data processing pipeline test completed successfully!")
