import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

base = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()

st.set_page_config(page_title="Ship Fuel Efficiency Predictor", layout="wide")

# --- Header / hero ---
st.title("🚢 Ship Fuel Efficiency Predictor 🚢 ")

# --- load artifacts ---
model_path = os.path.join(base, "artifacts", "best_model.pkl")
meta_path = os.path.join(base, "artifacts", "model_columns.json")

model = None
meta = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model from {model_path}: {e}")
else:
    st.error(f"Model not found at: {model_path}")

if os.path.exists(meta_path):
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        st.warning(f"Could not read {meta_path}: {e}")
        meta = None
else:
    st.warning(f"Metadata file not found at: {meta_path}")

# If meta is a list by mistake, try to recover to dict with keys we expect
if isinstance(meta, list):
    try:
        # try to find a dict element inside
        for item in meta:
            if isinstance(item, dict) and "numeric_cols" in item:
                meta = item
                break
        if not isinstance(meta, dict):
            meta = {
                "numeric_cols": ["distance", "fuel_consumption", "CO2_emissions"],
                "categorical_cols": ["ship_type"],
                "cat_values": {"ship_type": ["Fishing Trawler", "Oil Service Boat", "Surfer Boat", "Tanker Ship"]},
                "target": "engine_efficiency",
                "model_name": "auto-fixed"
            }
    except Exception:
        meta = {
            "numeric_cols": ["distance", "fuel_consumption", "CO2_emissions"],
            "categorical_cols": ["ship_type"],
            "cat_values": {"ship_type": ["Fishing Trawler", "Oil Service Boat", "Surfer Boat", "Tanker Ship"]},
            "target": "engine_efficiency",
            "model_name": "auto-fixed"
        }

if model is None or meta is None:
    st.stop()

numeric_cols = meta.get("numeric_cols", [])
categorical_cols = meta.get("categorical_cols", [])
cat_values = meta.get("cat_values", {})

# --- Image + Video area ---
local_names = ["ship.jpeg", "ship.jpg", "ship.png"]
img_path = None
for fn in local_names:
    p = os.path.join(base, fn)
    if os.path.exists(p):
        img_path = p
        break

if img_path:
    try:
        st.image(img_path, caption="Fuel-efficient ship", use_container_width=True)
    except Exception:
        st.warning("Found image but couldn't load it; continuing without image.")
        img_path = None

if not img_path:
    remote_img = "https://images.unsplash.com/photo-1542291026-7eec264c27ff?q=80&w=1400&auto=format&fit=crop&ixlib=rb-4.0.3&s=example"
    st.info("Local ship image not found — using a fallback online image.")
    st.image(remote_img, caption="Fuel-efficient ship (fallback)", use_container_width=True)

st.markdown("### Reference video")
yt_url = "https://www.youtube.com/watch?v=wQMx7wc4jh8"
st.video(yt_url)

st.success("Model and metadata loaded successfully!")

# --- Input form in sidebar ---
with st.sidebar.form("input_form"):
    st.header("Input parameters")
    defaults = {"distance": 100.0, "fuel_consumption": 100.0, "CO2_emissions": 100.0}
    inp = {}
    for col in numeric_cols:
        inp[col] = st.number_input(col.replace("_", " ").title(), value=float(defaults.get(col, 100.0)), step=1.0, format="%.2f")
    for col in categorical_cols:
        choices = cat_values.get(col, [])
        if not choices:
            choices = ["Unknown"]
        inp[col] = st.selectbox(col.replace("_", " ").title(), choices)
    submitted = st.form_submit_button("Predict Engine Efficiency")

# --- Prepare input dataframe ---
df_input = pd.DataFrame([inp])

# --- Prediction helper ---
def safe_predict(pipeline, df):
    try:
        pred = pipeline.predict(df)
        return float(pred[0])
    except Exception as e:
        raise

# --- Show raw input and predict on submit ---
if submitted:
    st.markdown("## Raw df_input:")
    st.dataframe(df_input)

    # Try prediction
    try:
        pred_val = safe_predict(model, df_input)
        st.metric("Predicted Engine Efficiency", f"{pred_val:.3f}")
    except Exception as e:
        st.error("Prediction failed. Attempting debug information below.")
        st.exception(e)
        # debugging info: try to show expected feature names if possible
        try:
            if hasattr(model, "named_steps") and "preproc" in model.named_steps:
                pre = model.named_steps["preproc"]
                try:
                    feat_names = pre.get_feature_names_out(df_input.columns)
                except Exception:
                    try:
                        feat_names = pre.get_feature_names_out()
                    except Exception:
                        feat_names = None
                st.write("Preprocessor feature names (preview):", feat_names)
        except Exception:
            pass
        st.stop()

    # --- Show model internals: intercept + coefficients if available ---
    intercept = None
    coefs = None
    feat_names = None
    try:
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            estimator = model.named_steps["model"]
            if hasattr(estimator, "intercept_"):
                intercept = float(estimator.intercept_)
            if hasattr(estimator, "coef_"):
                coefs = estimator.coef_.ravel()
            # attempt to get transformed feature names (ColumnTransformer)
            if "preproc" in model.named_steps:
                pre = model.named_steps["preproc"]
                try:
                    feat_names = pre.get_feature_names_out(df_input.columns)
                except Exception:
                    try:
                        feat_names = pre.get_feature_names_out()
                    except Exception:
                        feat_names = None
        else:
            # model may be a raw estimator
            estimator = model
            if hasattr(estimator, "intercept_"):
                intercept = float(estimator.intercept_)
            if hasattr(estimator, "coef_"):
                coefs = estimator.coef_.ravel()
            feat_names = list(df_input.columns)
    except Exception:
        intercept = None
        coefs = None
        feat_names = None

    st.markdown("## Regression Model Summary")
    if intercept is not None:
        st.markdown(f"**Intercept:** `{intercept:.6f}`")
    if coefs is not None and feat_names is not None and len(coefs) == len(feat_names):
        df_coef = pd.DataFrame({"feature": feat_names, "coef": coefs})
        st.table(df_coef)
        # build human-friendly equation string
        eq_parts = []
        for f, c in zip(feat_names, coefs):
            short = f.replace("num__", "").replace("cat__", "").replace("preproc__", "")
            eq_parts.append(f"({c:.3f} × {short})")
        equation = f"Engine Efficiency = {intercept:.3f} + " + " + ".join(eq_parts)
        st.markdown("**Equation:**")
        st.write(equation)
    else:
        st.info("Model coefficients not available or feature names mismatch; showing basic prediction only.")

    # show test of transform for debugging (optional)
    try:
        if hasattr(model, "named_steps") and "preproc" in model.named_steps:
            pre = model.named_steps["preproc"]
            transformed = pre.transform(df_input)
            st.markdown("### Transformed array (first row):")
            st.write(pd.DataFrame(transformed).iloc[0].round(4))
    except Exception:
        pass

    st.success("Done ✔")

else:
    st.info("Fill the parameters on the left and press Predict Engine Efficiency.")
