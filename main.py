import streamlit as st
import pandas as pd
import numpy as np
import shap
from tensorflow.keras.models import load_model

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import logging

logging.basicConfig(level=logging.ERROR)
st.title("Engine Maintenance")

file = st.file_uploader("Enter your data file")

binary_model = load_model("models/model.keras")
rul_model = load_model("models/model.keras")

sequence_length = 50
num_features = 24

def _compute_feature_importance(explainer, sample_values: np.ndarray, feature_names):
    """Return mean absolute SHAP values per feature for the provided sample."""
    shap_values = explainer(sample_values)
    values = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
    per_feature = np.mean(np.abs(values), axis=(0, 1))
    return per_feature

if file is not None:
    df = pd.read_csv(file)
    st.write(df)
    
    if df.shape != (sequence_length, num_features):
        st.error(f"Expected data with shape ({sequence_length}, {num_features}), but received {df.shape}.")
    else:
        data = df.to_numpy().reshape(1, sequence_length, num_features)

        # Binary classification prediction
        prediction = binary_model.predict(data)
        prediction = prediction[0][0]

        col1, col2 = st.columns(2)
        with col1:
            if round(prediction):
                st.title("The engine is not damaged")
            else:
                st.title("The engine is damaged")
                
        # Remaining Useful Life prediction
        with col2:
            RUL = rul_model.predict(data)
            st.title(f"The number of cycles left = {round(1 / RUL[0][0])}")

        with st.expander("See why the model predicted this"):
            st.caption("Feature importance based on SHAP values for the binary classifier.")
            # Use the current sample as the background for simplicity; replace with a curated set if available.
            explainer = shap.Explainer(binary_model, data)
            feature_names = df.columns.tolist()

            try:
                # Compute the feature importance from SHAP values
                feature_importance = _compute_feature_importance(explainer, data, feature_names)
                
                # Get top N most important features based on SHAP values
                top_n = 10 if len(feature_importance) > 10 else len(feature_importance)
                top_indices = np.argsort(feature_importance)[::-1][:top_n]

                # Display the most important features
                st.write("Top contributing features based on SHAP values:")
                for i in range(top_n):
                    feature_name = feature_names[top_indices[i]]
                    importance_value = feature_importance[top_indices[i]]
                    st.write(f"{feature_name}: {importance_value:.4f}")
                    
            except Exception as exc:  # Handle any SHAP computation errors
                st.warning(f"Could not compute SHAP explanations: {exc}")

