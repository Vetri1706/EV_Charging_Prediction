import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# === Page config ===
st.set_page_config(page_title="EV Adoption Predictor", layout="wide")

# === Utility / constants ===
EXPECTED_FEATURE_COLS = [
    "months_since_start",
    "county_encoded",
    "ev_total_lag1",
    "ev_total_lag2",
    "ev_total_lag3",
    "ev_total_roll_mean_3",
    "ev_total_pct_change_1",
    "ev_total_pct_change_3",
    "ev_growth_slope"
]

# === Load model safely ===
def load_model(filename="forecasting_ev_model.pkl"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, filename)
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Make sure '{filename}' is beside this script.")
        st.stop()
    try:
        model_obj = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    if not hasattr(model_obj, "predict"):
        st.error("Loaded object does not have `.predict()`; the model file is invalid or wrong.")
        st.stop()

    return model_obj

model = load_model()  # trained estimator

# === Styling ===
st.markdown(
    """
    <style>
        body { background-color: #fcf7f7; color: #000; }
        .stApp { background: linear-gradient(to right, #c2d3f2, #7f848a); }
        .big-title { font-size: 2.3rem; font-weight: 700; }
        .subtitle { font-size: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Header ===
st.markdown("<div class='big-title' style='text-align:center; color: white;'>🔮 EV Adoption Forecaster for a County in Washington State</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle' style='text-align:center; color: white; margin-bottom:15px;'>Welcome to the Electric Vehicle (EV) Adoption Forecast tool.</div>", unsafe_allow_html=True)

# Optional image (if present)
if os.path.exists("Copilot_20250802_104644.png"):
    st.image("Copilot_20250802_104644.png", use_container_width=True)

st.markdown("<div style='text-align:left; font-size:18px; color:white; margin-top:5px;'>Select a county and see forecasted EV adoption trend for next 3 years.</div>", unsafe_allow_html=True)

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# === Sidebar / inputs ===
st.sidebar.header("🔍 Options")
county_list = sorted(df["County"].dropna().unique().tolist())
selected_county = st.sidebar.selectbox("Select base County", county_list)
show_forecast = st.sidebar.checkbox("Show forecast chart", value=True)
compare_counties = st.sidebar.multiselect("Compare up to 3 counties", county_list, max_selections=3)

# === Forecast helper ===
def make_forecast(df_slice, county_code, horizon=36, base_cumulative=0):
    hist_ev = list(df_slice["Electric Vehicle (EV) Total"].values[-6:])
    cum_ev = list(np.cumsum(hist_ev))
    months_since = df_slice["months_since_start"].max()
    last_date = df_slice["Date"].max()

    future = []
    for i in range(1, horizon + 1):
        months_since += 1
        forecast_date = last_date + pd.DateOffset(months=i)
        lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cum = cum_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0

        feature_row = {
            "months_since_start": months_since,
            "county_encoded": county_code,
            "ev_total_lag1": lag1,
            "ev_total_lag2": lag2,
            "ev_total_lag3": lag3,
            "ev_total_roll_mean_3": roll_mean,
            "ev_total_pct_change_1": pct_change_1,
            "ev_total_pct_change_3": pct_change_3,
            "ev_growth_slope": ev_growth_slope
        }

        sample_df = pd.DataFrame([feature_row])
        # ensure column order and presence
        try:
            sample_df = sample_df[EXPECTED_FEATURE_COLS]
        except KeyError as ke:
            st.error(f"Feature columns mismatch: {ke}")
            st.stop()

        pred = model.predict(sample_df)[0]
        future.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

        # roll history
        hist_ev.append(pred)
        if len(hist_ev) > 6:
            hist_ev.pop(0)
        cum_ev.append(cum_ev[-1] + pred)
        if len(cum_ev) > 6:
            cum_ev.pop(0)

    forecast_df = pd.DataFrame(future)
    forecast_df["Cumulative EV"] = forecast_df["Predicted EV Total"].cumsum() + base_cumulative
    return forecast_df

# === Single county forecast ===
county_df = df[df["County"] == selected_county].sort_values("Date")
if county_df.empty:
    st.warning(f"No data for county '{selected_county}'.")
    st.stop()

county_code = county_df["county_encoded"].iloc[0]
historical_cum = county_df[["Date", "Electric Vehicle (EV) Total"]].copy()
historical_cum["Cumulative EV"] = historical_cum["Electric Vehicle (EV) Total"].cumsum()
historical_cum["Source"] = "Historical"

forecast_df = make_forecast(
    df_slice=county_df,
    county_code=county_code,
    horizon=36,
    base_cumulative=historical_cum["Cumulative EV"].iloc[-1]
)
forecast_df["Source"] = "Forecast"

# === Combined for plotting ===
combined = pd.concat([
    historical_cum[["Date", "Cumulative EV", "Source"]],
    forecast_df[["Date", "Cumulative EV", "Source"]]
], ignore_index=True)

# === Plot cumulative for selected county ===
st.subheader(f"📊 Cumulative EV Forecast for {selected_county} County")
if show_forecast:
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, data in combined.groupby("Source"):
        ax.plot(data["Date"], data["Cumulative EV"], label=label, marker="o")
    ax.set_title(f"Cumulative EV Trend - {selected_county} (3-Year Forecast)", fontsize=14, color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Cumulative EV Count", color="white")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor("#1c1c1c")
    ax.tick_params(colors="white")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Forecast chart is hidden. Toggle 'Show forecast chart' in the sidebar to view.")

# === Growth summary single county ===
historical_total = historical_cum["Cumulative EV"].iloc[-1]
forecasted_total = forecast_df["Cumulative EV"].iloc[-1]
if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.success(f"Based on the forecast, EV adoption in **{selected_county}** shows a **{trend} of {forecast_growth_pct:.2f}%** over 3 years.")
else:
    st.warning("Historical cumulative EV is zero; cannot compute growth percentage.")

# === Multi-county comparison ===
if compare_counties:
    st.markdown("---")
    st.header("Compare EV Adoption Trends for Selected Counties")
    comparison_frames = []
    growth_summaries = []

    for cty in compare_counties:
        cty_df = df[df["County"] == cty].sort_values("Date")
        if cty_df.empty:
            continue
        cty_code = cty_df["county_encoded"].iloc[0]
        hist_cum_cty = cty_df[["Date", "Electric Vehicle (EV) Total"]].copy()
        hist_cum_cty["Cumulative EV"] = hist_cum_cty["Electric Vehicle (EV) Total"].cumsum()

        fc_df_cty = make_forecast(
            df_slice=cty_df,
            county_code=cty_code,
            horizon=36,
            base_cumulative=hist_cum_cty["Cumulative EV"].iloc[-1]
        )
        # combine
        full_cty = pd.concat([
            hist_cum_cty[["Date", "Cumulative EV"]].assign(Source="Historical"),
            fc_df_cty[["Date", "Cumulative EV"]].assign(Source="Forecast")
        ], ignore_index=True)
        full_cty["County"] = cty
        comparison_frames.append(full_cty)

        # growth
        hist_total = hist_cum_cty["Cumulative EV"].iloc[-1]
        fc_total = fc_df_cty["Cumulative EV"].iloc[-1]
        if hist_total > 0:
            growth_pct = ((fc_total - hist_total) / hist_total) * 100
            growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
        else:
            growth_summaries.append(f"{cty}: N/A")

    if comparison_frames:
        comp_df = pd.concat(comparison_frames, ignore_index=True)

        # Plot comparison
        st.subheader("📈 Comparison: Historical + Forecasted Cumulative EV Adoption")
        fig, ax = plt.subplots(figsize=(14, 7))
        for cty, grp in comp_df.groupby("County"):
            ax.plot(grp["Date"], grp["Cumulative EV"], marker="o", label=cty)
        ax.set_title("EV Adoption Trends Across Counties (3-Year Forecast)", fontsize=16, color="white")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Cumulative EV Count", color="white")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1c1c1c")
        fig.patch.set_facecolor("#1c1c1c")
        ax.tick_params(colors="white")
        ax.legend(title="County")
        st.pyplot(fig)

        st.success("Forecasted EV adoption growth over next 3 years — " + " | ".join(growth_summaries))

# === Download forecast for base county ===
st.markdown("---")
st.download_button(
    label="📥 Download Forecast CSV",
    data=forecast_df.to_csv(index=False),
    file_name=f"{selected_county}_ev_forecast.csv",
    mime="text/csv"
)

st.markdown("Prepared for the **AICTE Internship Cycle 2 by S4F**")
