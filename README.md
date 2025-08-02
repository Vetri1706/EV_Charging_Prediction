# EV Charging Adoption Forecast – Internship Task

## 📘 Problem Statement
Forecast county-level electric vehicle (EV) adoption for the next 3 years using historical data to support regional planning and policymaking.

## 🧠 Model
- **Type:** Random Forest Regressor (scikit-learn) serialized with `joblib`.  
- **Features:** Lag EV totals, 3-month rolling mean, percentage changes, and trend slope based on recent cumulative growth.  
- **Forecast Horizon:** 36 months per county, with cumulative prediction and growth percentage insights.  
- **Comparison:** Supports multi-county cumulative trend comparison.

## 📁 Files Included
1. `app.py` – Streamlit application integrating data, model loading, forecasting logic, visualization, and export.  
2. `forecasting_ev_model.pkl` – Trained & serialized regression model.  
3. `preprocessed_ev_data.csv` – Cleaned historical EV adoption data with required columns (`Date`, `County`, `county_encoded`, `months_since_start`, `Electric Vehicle (EV) Total`, etc.).  
4. `requirements.txt` – Pinned dependencies for reproducible environment.  
5. `README.md` – Project documentation (includes setup, usage, methodology, objectives).  
6. `ev-car-factory.jpg` *(optional)* – UI/banner image for the dashboard.  



## How to Run
1. 
1. **Clone the repository**
   ```bash

   git clone https://github.com/Vetri1706/EV-Adoption-Forecast.git
   cd EV-Adoption-Forecast
# Activate on Windows
venv\Scripts\activate
# Activate on Mac/Linux
source venv/bin/activate
# Run
streamlit run app.py
## ▶️ Usage

1. After launching, select a county from the sidebar to view its historical and 3-year forecasted EV adoption.  
2. Optionally pick up to 3 counties to compare cumulative trends and growth percentages.  
3. Toggle the forecast chart visibility and download the forecast CSV for offline analysis.  
4. Insights like percentage growth are shown automatically beneath the chart.

## ⚙️ Configuration / Notes

- Ensure `preprocessed_ev_data.csv` contains columns: `Date`, `County`, `county_encoded`, `months_since_start`, and `Electric Vehicle (EV) Total`.  
- The model file `forecasting_ev_model.pkl` must be a trained scikit-learn regressor saved with `joblib.dump(...)`.  
- Column order/feature engineering in the app must match how the model was trained (lags, roll mean, pct changes, slope).

## 🛠 Troubleshooting

- **Model not found**: Verify `forecasting_ev_model.pkl` is in the same directory as `app.py` and loaded via the robust path logic.  
- **Invalid model / no `.predict()`**: Replace with a properly serialized trained estimator (`joblib.dump(trained_model, ...)`).  
- **Feature mismatch errors**: Ensure the app’s engineered feature names align with what the model expects (see `EXPECTED_FEATURE_COLS`).  
- **Slow load**: Enable caching (`@st.cache_data`) for data and avoid retraining inside the app.

## 📦 requirements.txt snippet
- pandas
- numpy
- scikit-learn
- joblib
- streamlit
- matplotlib






