#!/usr/bin/env python3
"""
Quick demo script for the Exoplanet AI Detector.
Shows how to use the model programmatically.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import ExoplanetDetector
from data_processor import ExoplanetDataProcessor, create_sample_light_curve_data
import pandas as pd

def main():
    """Run a quick demo of the exoplanet detector."""
    print("🪐 Exoplanet AI Detector - Quick Demo")
    print("=" * 50)
    
    # Load the trained model
    detector = ExoplanetDetector()
    detector.load_model('models/exoplanet_detector.pkl')
    print("✅ Model loaded successfully")
    
    # Demo predictions
    print("\n🔍 Testing with sample data...")
    
    # Test case 1: Likely planet
    print("\nTest Case 1: Strong planet candidate")
    features1 = {
        'period': 15.0,      # 15-day orbit
        'duration': 0.12,    # 2.9-hour transit
        'depth': 0.025,      # 2.5% depth
        'snr': 20.0,         # High signal-to-noise
        'impact_param': 0.3, # Low impact parameter
        'eccentricity': 0.05, # Low eccentricity
        'stellar_radius': 1.1, # Solar-like star
        'stellar_mass': 1.0,   # Solar mass
        'stellar_teff': 5800   # Solar temperature
    }
    
    pred1, conf1 = detector.predict_single(features1)
    status1 = "🪐 PLANET DETECTED!" if pred1 == 1 else "❌ NO PLANET"
    print(f"   Result: {status1} (Confidence: {conf1:.1%})")
    
    # Test case 2: Unlikely planet
    print("\nTest Case 2: Weak planet candidate")
    features2 = {
        'period': 2.0,       # Very short orbit
        'duration': 0.05,    # Very short transit
        'depth': 0.002,      # Very shallow depth
        'snr': 3.0,          # Low signal-to-noise
        'impact_param': 0.8, # High impact parameter
        'eccentricity': 0.3, # High eccentricity
        'stellar_radius': 0.5, # Small star
        'stellar_mass': 0.3,   # Low mass star
        'stellar_teff': 4000   # Cool star
    }
    
    pred2, conf2 = detector.predict_single(features2)
    status2 = "🪐 PLANET DETECTED!" if pred2 == 1 else "❌ NO PLANET"
    print(f"   Result: {status2} (Confidence: {conf2:.1%})")
    
    # Test case 3: Borderline case
    print("\nTest Case 3: Borderline candidate")
    features3 = {
        'period': 8.0,       # Medium orbit
        'duration': 0.08,    # Medium transit
        'depth': 0.012,      # Medium depth
        'snr': 8.0,          # Medium signal-to-noise
        'impact_param': 0.6, # Medium impact parameter
        'eccentricity': 0.15, # Medium eccentricity
        'stellar_radius': 0.9, # Solar-like star
        'stellar_mass': 0.9,   # Solar mass
        'stellar_teff': 5500   # Solar temperature
    }
    
    pred3, conf3 = detector.predict_single(features3)
    status3 = "🪐 PLANET DETECTED!" if pred3 == 1 else "❌ NO PLANET"
    print(f"   Result: {status3} (Confidence: {conf3:.1%})")
    
    # Show feature importance
    print("\n📊 Top 5 Most Important Features:")
    importance_df = detector.get_feature_importance()
    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
        print(f"   {i}. {row['feature']}: {row['importance']:.3f}")
    
    # Load sample data
    print("\n📈 Sample Data Overview:")
    processor = ExoplanetDataProcessor()
    df = processor.load_data()
    print(f"   Total samples: {len(df)}")
    print(f"   Confirmed planets: {df['is_planet'].sum()}")
    print(f"   Planet detection rate: {df['is_planet'].mean():.1%}")
    
    print("\n🎉 Demo completed successfully!")
    print("🚀 Run 'streamlit run app.py' to launch the full web interface")

if __name__ == "__main__":
    main()
