#!/usr/bin/env python3
"""
Training script for the Exoplanet AI Detector.
Run this script to train the model before using the web app.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import train_exoplanet_model

def main():
    """Train the exoplanet detection model."""
    print("🚀 Starting Exoplanet AI Detector Training")
    print("=" * 50)
    
    try:
        # Train the model
        detector = train_exoplanet_model()
        
        print("\n✅ Training completed successfully!")
        print("🎯 You can now run the web app with: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
