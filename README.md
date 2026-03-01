# 🌍 Asteroid Risk Analyzer

A machine learning portfolio project for classifying Near-Earth Objects (NEOs) as **Potentially Hazardous** or **Safe** using NASA data and a Random Forest Classifier.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```
---

## 📁 Project Structure

```
asteroid_risk_analyzer/
├── app.py                     # Streamlit application
├── requirements.txt
├── README.md
├── data/
│   └── neos.csv               # Fetched asteroid data
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis
└── src/
    ├── __init__.py
    ├── data_loader.py         # NASA API fetcher + CSV loader
    ├── preprocessing.py       # Cleaning, feature engineering
    └── model.py               # Model training + evaluation
```

---

## 📊 Dataset Description

Data sourced from **NASA NeoWs (Near Earth Object Web Service)**:
- Asteroid physical properties (size, magnitude)
- Close approach data (velocity, miss distance)
- Hazard classification labels

### Features Used

| Feature | Description |
|---|---|
| `absolute_magnitude_h` | Intrinsic brightness (proxy for size) |
| `diameter_min_km` | Minimum estimated diameter |
| `diameter_max_km` | Maximum estimated diameter |
| `diameter_mean_km` | Derived: average diameter |
| `relative_velocity_km_s` | Speed relative to Earth at approach |
| `miss_distance_km` | Closest approach distance |
| `orbital_eccentricity` | Orbit shape (0=circle, 1=parabola) |

**Target:** `is_potentially_hazardous_asteroid` (binary: 0 = Safe, 1 = Hazardous)

---

## 🤖 Model Choice: Random Forest

**Why Random Forest?**
- Handles non-linear relationships between features
- Naturally resistant to overfitting with many trees
- Provides feature importance rankings
- Works well with mixed-scale features (no normalization needed)
- `class_weight='balanced'` handles the class imbalance (~18% hazardous)

**Training Configuration:**
- 200 estimators, max depth 10
- Stratified 80/20 train/test split
- `class_weight='balanced'` to handle class imbalance

---

## 📈 Evaluation Metrics

Given the class imbalance and safety-critical context, **Recall** for the hazardous class is prioritized:
- **Recall** — catch as many real hazards as possible
- **F1-Score** — balance precision and recall
- **ROC-AUC** — overall discriminative ability

---

## 💡 Key Insights

1. **Miss distance** is the strongest single predictor of hazard classification
2. **Absolute magnitude** and **diameter** are highly correlated
3. **Orbital eccentricity** helps distinguish crossing orbits from circular ones
4. The dataset has ~18% hazardous asteroids — class weighting is essential

---

## 📡 Data Source

- **API:** [NASA NeoWs](https://api.nasa.gov/) — free key at api.nasa.gov
- **Documentation:** [NeoWs Docs](https://api.nasa.gov/api.html#NeoWS)
