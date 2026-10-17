from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="AI House Price Predictor", page_icon="🏡", layout="centered")

st.title("🏡 AI House Price Predictor")
st.write("Enter the details of a house to estimate its price using a machine-learning model trained on the project's dataset.")
st.caption("Educational prototype. Estimates depend on the size and quality of the dataset and are not official property valuations.")

DATA_FILES = [
    Path("house_price_dataset.csv"),
    Path("house_price_dataset.xlsx"),
    Path("house_price_dataset.csv..xlsx"),
]


def load_dataset():
    for path in DATA_FILES:
        if path.exists():
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path), path.name
            return pd.read_excel(path), path.name
    return None, None


def clean_numeric(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    # Allows values such as "₹ 45,00,000" or "45.5" to be interpreted as numbers.
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


df, source_name = load_dataset()
if df is None:
    st.error("Dataset file not found. Upload your final dataset to the same GitHub repository as this app.")
    st.info("Use the filename house_price_dataset.csv or house_price_dataset.xlsx. CSV is recommended.")
    st.stop()

# Make column names reliable for display but keep the original column names for lookup.
df.columns = [str(c).strip() for c in df.columns]
price_candidates = [
    c for c in df.columns
    if any(word in c.lower() for word in ("price", "cost", "amount", "value"))
]
if not price_candidates:
    st.error("I couldn't automatically identify the price column. Rename the target column to include the word 'price', then upload the file again.")
    st.write("Columns found:", list(df.columns))
    st.stop()

preferred = next((c for c in price_candidates if "price_in_lakhs" in c.lower()), price_candidates[0])
target = st.selectbox("Which column contains the house price?", price_candidates, index=price_candidates.index(preferred))

# Use numeric features (area, bedrooms, bathrooms, balconies, etc.) and exclude ID-like columns.
feature_cols = []
for col in df.columns:
    if col == target or any(word in col.lower() for word in ("id", "serial", "index")):
        continue
    numeric = clean_numeric(df[col])
    if numeric.notna().sum() >= max(3, int(len(df) * 0.5)) and numeric.nunique(dropna=True) > 1:
        feature_cols.append(col)

if not feature_cols:
    st.error("No usable numeric house-feature columns were found. Make sure area, bedrooms, bathrooms, and balconies are stored as numbers.")
    st.write("Columns found:", list(df.columns))
    st.stop()

work = pd.DataFrame({col: clean_numeric(df[col]) for col in feature_cols})
work[target] = clean_numeric(df[target])
work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
work = work.dropna(axis=1, how="all")
feature_cols = [col for col in feature_cols if col in work.columns]
if len(work) < 5 or not feature_cols:
    st.error("The dataset needs at least 5 usable rows with a numeric price and numeric house features.")
    st.stop()

X = work[feature_cols].copy()
X = X.fillna(X.median(numeric_only=True)).fillna(0)
y = work[target]

# Report a hold-out evaluation when possible, then train the final model on all available rows.
model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=1)
if len(work) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    evaluation_model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=1)
    evaluation_model.fit(X_train, y_train)
    y_pred = evaluation_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    try:
        score = r2_score(y_test, y_pred)
    except Exception:
        score = float("nan")
else:
    mae = None
    score = None
model.fit(X, y)

st.success(f"Model ready — trained on {len(work)} usable rows from {source_name}.")
st.subheader("Enter house details")
values = {}
left, right = st.columns(2)
for i, col in enumerate(feature_cols):
    series = X[col]
    min_value = float(series.min())
    max_value = float(series.max())
    median_value = float(series.median())
    span = max(max_value - min_value, abs(max_value) * 0.1, 1.0)
    low = min(0.0, min_value) if min_value >= 0 else min_value - span * 0.1
    high = max_value + span * 0.25
    step = 1.0 if all(float(v).is_integer() for v in series.dropna().head(50)) else 0.1
    container = left if i % 2 == 0 else right
    with container:
        values[col] = st.number_input(
            col.replace("_", " ").title(),
            min_value=float(low),
            max_value=float(high),
            value=float(median_value),
            step=float(step),
            help=f"Dataset range: {min_value:g} to {max_value:g}",
        )

if st.button("Predict House Price", type="primary", use_container_width=True):
    input_row = pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)
    prediction = float(model.predict(input_row)[0])
    if "lakh" in target.lower():
        st.metric("Estimated House Price", f"₹ {prediction:,.2f} lakh")
        st.caption(f"Approximately ₹ {prediction * 100000:,.0f}")
    else:
        st.metric("Estimated House Price", f"₹ {prediction:,.0f}")
    st.caption("This estimate is generated by a Random Forest model trained from the dataset included with this app.")

with st.expander("Model and dataset details"):
    st.write("**Target column:**", target)
    st.write("**Input features:**", feature_cols)
    st.write("**Rows used:**", len(work))
    if mae is not None:
        st.write(f"**Hold-out mean absolute error:** {mae:,.3f} (in the target column's units)")
        st.write(f"**Hold-out R² score:** {score:.3f}" if np.isfinite(score) else "**Hold-out R² score:** not available")
    st.write("The app re-trains the model from the included dataset when it starts; it does not load a separate model file from Colab.")
