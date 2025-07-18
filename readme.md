# ⚡ EV Energy Forecasting Project

This project focuses on forecasting the adoption of electric vehicles (EVs) across counties using historical vehicle registration data. The aim is to help stakeholders understand and prepare for the future growth of electric mobility.

---

##  Dataset Information

**Source**: Kaggle  
**File**: `Electric_Vehicle_Population_By_County.csv`  
**Columns**:

- `Date` — Date of data collection  
- `County`, `State` — Geographical information  
- `Vehicle Primary Use` — Commercial or Personal  
- `Battery Electric Vehicles (BEVs)`  
- `Plug-In Hybrid Electric Vehicles (PHEVs)`  
- `Electric Vehicle (EV) Total` *(Target variable)*  
- `Non-Electric Vehicle Total`  
- `Total Vehicles`  
- `Percent Electric Vehicles`

---

## Objective

To predict the **`Electric Vehicle (EV) Total`** in different counties based on historical data, helping to forecast EV adoption trends.

---

## Technologies Used

- **Python**
- **Pandas, NumPy** — Data processing
- **Matplotlib, Seaborn** — Data visualization
- **Scikit-learn** — Machine Learning models (Linear Regression, Decision Tree, etc.)
- **Jupyter Notebook** — Development environment
- **Anaconda** — Package management

---

## ML Workflow

1. **Data Cleaning & Preprocessing**
   - Null value handling
   - Encoding categorical variables
   - Date parsing

2. **Feature Selection**
   - Independent variables (X): County, State, Date, Vehicle Use, etc.
   - Dependent variable (y): `Electric Vehicle (EV) Total`

3. **Model Building**
   - Split data using `train_test_split`
   - Apply regression models
   - Evaluate using R² Score, MAE, MSE

4. **Result Visualization**
   - Line plots
   - Bar charts
   - Prediction vs Actual graphs

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/ev-forecast.git
   cd ev-forecast
