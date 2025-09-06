# 🪐 Exoplanet AI Detector

A machine learning-powered tool for detecting exoplanets from NASA's telescope data. This project uses AI to analyze light curve features and stellar parameters to identify potential planetary candidates with high accuracy.

## 🚀 Features

- **AI-Powered Detection**: Random Forest model trained on exoplanet data
- **Interactive Web Interface**: Beautiful Streamlit dashboard for predictions
- **Real-time Visualization**: Interactive light curve plots and feature analysis
- **Confidence Scoring**: Probability-based predictions with confidence intervals
- **Feature Importance**: Understand which parameters matter most for detection

## 🎯 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

### 3. Launch the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔬 How It Works

1. **Input Parameters**: Enter stellar system characteristics (period, depth, SNR, etc.)
2. **AI Analysis**: Our trained model analyzes the features using Random Forest
3. **Prediction**: Get a binary classification (Planet/No Planet) with confidence score
4. **Visualization**: See simulated light curves and feature importance plots

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 9 key parameters (orbital period, transit depth, stellar properties, etc.)
- **Training Data**: Synthetic data mimicking NASA Kepler observations
- **Performance**: Balanced accuracy with class weight handling

## 🛠️ Project Structure

```
exoplanet-ai-detector/
├── app.py                 # Main Streamlit web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── src/
│   ├── data_processor.py  # Data preprocessing pipeline
│   └── model.py          # ML model implementation
├── data/                 # Training data (generated)
├── models/               # Trained model files
└── utils/                # Utility functions
```

## 🎨 Demo Features

### Prediction Interface
- Interactive parameter input form
- Real-time prediction with confidence scores
- Visual feedback with color-coded results

### Data Explorer
- Sample data visualization
- Feature distribution plots
- Model performance metrics

### Model Information
- Feature importance rankings
- Model architecture details
- Training statistics

## 🌟 Inspiration

This project is inspired by:
- **ExoMiner**: NASA's AI model that identified 301 exoplanets
- **Planet Hunters**: Citizen science platform for exoplanet discovery
- **Kepler Mission**: NASA's planet-hunting telescope data

## 🔮 Future Enhancements

- Real NASA dataset integration
- Deep learning models (CNNs for light curves)
- Multi-class classification (planet types)
- Real-time data processing
- API for research integration

## 📈 Technical Stack

- **Backend**: Python, scikit-learn, pandas, numpy
- **Frontend**: Streamlit, Plotly
- **Data**: NASA Kepler/TESS datasets (synthetic for demo)
- **ML**: Random Forest Classifier with feature engineering

## 🤝 Contributing

This is a hackathon demo project. Feel free to fork and extend!

## 📄 License

MIT License - Built for the astronomy community

---

**Built with ❤️ for the astronomy community**  
🚀 Powered by NASA data and AI technology
