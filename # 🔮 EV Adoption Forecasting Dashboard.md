# 🔮 EV Adoption Forecasting Dashboard

An interactive Streamlit web application that forecasts county-level Electric Vehicle (EV) adoption for the next 3 years using historical data and machine learning.  
This project was developed as part of **AICTE Internship Cycle 2**.

---

## 📌 Features
- **County Selection** – View historical & forecasted EV adoption trends for any county.
- **3-Year Forecast** – Predict cumulative EV adoption using a trained regression model.
- **Multi-County Comparison** – Compare adoption growth across up to 3 counties.
- **Growth Insights** – Display forecasted growth percentages.
- **Downloadable Forecast** – Export forecast data as CSV.

---

## 🛠 Tools & Technologies
1. **Python** – Core programming language.
2. **Pandas & NumPy** – Data preprocessing & feature engineering.
3. **scikit-learn & Joblib** – Machine learning model training and persistence.
4. **Streamlit & Matplotlib** – Interactive UI & visualization.
5. **OS & Caching** – File handling and `st.cache_data` for performance.

---

## 📊 Methodology
1. **Data Collection** – Load historical EV adoption data.
2. **Feature Engineering** – Create lag, rolling average, and growth rate features.
3. **Model Training** – Train and validate a Random Forest Regressor model.
4. **Forecasting** – Predict EV adoption trends for the next 36 months.
5. **Deployment** – Build an interactive Streamlit dashboard with export options.

---

## 🎯 Learning Objectives
1. Perform data preprocessing & feature engineering for time-series forecasting.
2. Train and deploy a machine learning regression model.
3. Visualize historical vs forecasted trends interactively.
4. Enable multi-county comparisons with growth metrics.
5. Build a robust, user-friendly web app.

---

## 📂 Project Structure
