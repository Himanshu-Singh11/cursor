"""
Exoplanet AI Detector - Streamlit Web App
A demo interface for detecting exoplanets using machine learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import ExoplanetDataProcessor, create_sample_light_curve_data
from model import ExoplanetDetector

# Page configuration
st.set_page_config(
    page_title="Exoplanet AI Detector",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .nasa-logo {
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .planet-detected {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .no-planet {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model."""
    try:
        detector = ExoplanetDetector()
        model_path = "models/exoplanet_detector.pkl"
        if os.path.exists(model_path):
            detector.load_model(model_path)
            return detector
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    processor = ExoplanetDataProcessor()
    return processor.load_data()

def create_light_curve_plot(light_curve_data, prediction=None, confidence=None):
    """Create an interactive light curve plot."""
    fig = go.Figure()
    
    # Add light curve
    fig.add_trace(go.Scatter(
        x=light_curve_data['time'],
        y=light_curve_data['flux'],
        mode='lines+markers',
        name='Light Curve',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # Add transit region highlight
    transit_center = light_curve_data['transit_center']
    transit_width = light_curve_data['transit_width']
    
    fig.add_vrect(
        x0=transit_center - transit_width/2,
        x1=transit_center + transit_width/2,
        fillcolor="red",
        opacity=0.2,
        annotation_text="Transit Region" if prediction == 1 else "Potential Transit",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title="Light Curve Analysis",
        xaxis_title="Time (days)",
        yaxis_title="Relative Flux",
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    # Add prediction info to title if available
    if prediction is not None and confidence is not None:
        status = "PLANET DETECTED" if prediction == 1 else "NO PLANET"
        color = "green" if prediction == 1 else "red"
        fig.update_layout(
            title=f"Light Curve Analysis - {status} (Confidence: {confidence:.1%})",
            title_font_color=color
        )
    
    return fig

def create_feature_importance_plot(importance_df):
    """Create feature importance plot."""
    fig = px.bar(
        importance_df.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Most Important Features",
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="nasa-logo">🚀 NASA 🚀</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">Exoplanet AI Detector</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        Discover new worlds using artificial intelligence and NASA's telescope data
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Controls")
    
    # Load model
    detector = load_model()
    
    if detector is None:
        st.error("⚠️ Model not found! Please train the model first by running: `python src/model.py`")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📊 Model Info", "📈 Data Explorer", "ℹ️ About"])
    
    with tab1:
        st.header("Exoplanet Prediction")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Stellar System Parameters")
            
            # Create input form
            with st.form("prediction_form"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=1000.0, value=10.0, step=0.1)
                    duration = st.number_input("Transit Duration (days)", min_value=0.01, max_value=10.0, value=0.1, step=0.01)
                    depth = st.number_input("Transit Depth", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
                    snr = st.number_input("Signal-to-Noise Ratio", min_value=1.0, max_value=100.0, value=10.0, step=0.1)
                    impact_param = st.number_input("Impact Parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                
                with col_b:
                    eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=0.9, value=0.1, step=0.01)
                    stellar_radius = st.number_input("Stellar Radius (Solar radii)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                    stellar_mass = st.number_input("Stellar Mass (Solar masses)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
                    stellar_teff = st.number_input("Stellar Temperature (K)", min_value=2000, max_value=10000, value=5778, step=10)
                
                submitted = st.form_submit_button("🔍 Analyze for Exoplanet", use_container_width=True)
        
        with col2:
            st.subheader("Quick Demo")
            if st.button("🎲 Random Sample", use_container_width=True):
                # Generate random sample
                np.random.seed()
                period = np.random.exponential(10)
                duration = np.random.normal(0.1, 0.05)
                depth = np.random.exponential(0.01)
                snr = np.random.normal(10, 5)
                impact_param = np.random.uniform(0, 1)
                eccentricity = np.random.beta(2, 5)
                stellar_radius = np.random.normal(1, 0.3)
                stellar_mass = np.random.normal(1, 0.2)
                stellar_teff = np.random.normal(5778, 1000)
                
                st.rerun()
        
        # Prediction section
        if submitted:
            # Prepare input data
            input_features = {
                'period': period,
                'duration': duration,
                'depth': depth,
                'snr': snr,
                'impact_param': impact_param,
                'eccentricity': eccentricity,
                'stellar_radius': stellar_radius,
                'stellar_mass': stellar_mass,
                'stellar_teff': stellar_teff
            }
            
            # Make prediction
            prediction, confidence = detector.predict_single(input_features)
            
            # Display results
            st.markdown("---")
            
            # Prediction card
            prediction_class = "planet-detected" if prediction == 1 else "no-planet"
            status = "🪐 PLANET DETECTED!" if prediction == 1 else "❌ NO PLANET"
            
            st.markdown(f"""
            <div class="prediction-card {prediction_class}">
                <h2>{status}</h2>
                <h3>Confidence: {confidence:.1%}</h3>
                <p>Based on the input parameters, our AI model predicts this system 
                {'contains an exoplanet' if prediction == 1 else 'does not contain an exoplanet'}.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Light curve visualization
            st.subheader("Light Curve Simulation")
            light_curve_data = create_sample_light_curve_data()
            fig = create_light_curve_plot(light_curve_data, prediction, confidence)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature analysis
            st.subheader("Feature Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Signal-to-Noise", f"{snr:.1f}", "High" if snr > 15 else "Medium" if snr > 8 else "Low")
            with col2:
                st.metric("Transit Depth", f"{depth:.3f}", "Significant" if depth > 0.02 else "Moderate" if depth > 0.01 else "Shallow")
            with col3:
                st.metric("Orbital Period", f"{period:.1f} days", "Short" if period < 10 else "Medium" if period < 50 else "Long")
    
    with tab2:
        st.header("Model Information")
        
        # Model metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", "Random Forest", "Ensemble Method")
        with col2:
            st.metric("Features", "9", "Stellar & Orbital")
        with col3:
            st.metric("Training Samples", "800", "Balanced Dataset")
        with col4:
            st.metric("Model Status", "✅ Trained", "Ready for Prediction")
        
        # Feature importance
        st.subheader("Feature Importance")
        importance_df = detector.get_feature_importance()
        fig = create_feature_importance_plot(importance_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        st.subheader("Model Architecture")
        st.info("""
        **Random Forest Classifier:**
        - 100 decision trees
        - Balanced class weights for handling imbalanced data
        - Cross-validated for robustness
        - Feature importance ranking for interpretability
        """)
    
    with tab3:
        st.header("Data Explorer")
        
        # Load sample data
        df = load_sample_data()
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Confirmed Planets", df['is_planet'].sum())
        with col3:
            st.metric("Planet Rate", f"{df['is_planet'].mean():.1%}")
        with col4:
            st.metric("Features", len(df.columns) - 1)
        
        # Data distribution
        st.subheader("Data Distribution")
        
        # Create distribution plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Period Distribution', 'SNR Distribution', 'Depth Distribution', 'Temperature Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Period
        fig.add_trace(go.Histogram(x=df['period'], name='Period', nbinsx=30), row=1, col=1)
        # SNR
        fig.add_trace(go.Histogram(x=df['snr'], name='SNR', nbinsx=30), row=1, col=2)
        # Depth
        fig.add_trace(go.Histogram(x=df['depth'], name='Depth', nbinsx=30), row=2, col=1)
        # Temperature
        fig.add_trace(go.Histogram(x=df['stellar_teff'], name='Temperature', nbinsx=30), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab4:
        st.header("About Exoplanet AI Detector")
        
        st.markdown("""
        ## 🚀 Mission Statement
        
        The Exoplanet AI Detector leverages machine learning to accelerate the discovery of exoplanets 
        from NASA's telescope data. By analyzing light curve features and stellar parameters, our AI 
        can identify potential planetary candidates with high accuracy.
        
        ## 🔬 How It Works
        
        1. **Data Input**: Stellar system parameters and light curve features
        2. **AI Analysis**: Random Forest model trained on confirmed exoplanet data
        3. **Prediction**: Binary classification with confidence scores
        4. **Visualization**: Interactive light curve plots and feature analysis
        
        ## 📊 Model Performance
        
        - **Accuracy**: Trained on synthetic data mimicking real Kepler observations
        - **Features**: 9 key parameters including orbital period, transit depth, and stellar properties
        - **Method**: Ensemble Random Forest with balanced class weights
        
        ## 🌟 Inspiration
        
        This project is inspired by:
        - **ExoMiner**: NASA's AI model that identified 301 exoplanets
        - **Planet Hunters**: Citizen science platform for exoplanet discovery
        - **Kepler Mission**: NASA's planet-hunting telescope data
        
        ## 🛠️ Technical Stack
        
        - **Backend**: Python, scikit-learn, pandas
        - **Frontend**: Streamlit, Plotly
        - **Data**: NASA Kepler/TESS datasets (synthetic for demo)
        - **ML**: Random Forest Classifier
        
        ## 🎯 Future Enhancements
        
        - Real NASA dataset integration
        - Deep learning models (CNNs for light curves)
        - Multi-class classification (planet types)
        - Real-time data processing
        - API for research integration
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>Built with ❤️ for the astronomy community</p>
            <p>🚀 Powered by NASA data and AI technology</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
