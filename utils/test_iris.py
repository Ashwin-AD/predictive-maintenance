import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset (For example: Iris dataset)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train a simple model (Random Forest)
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Create an explainer object
explainer = shap.TreeExplainer(model)  # For tree-based models like RandomForest

# Compute SHAP values for the model's predictions on a subset of data
shap_values = explainer.shap_values(X)

# For multi-class classification, shap_values will be a list of SHAP values for each class
# Here, we take the SHAP values for the first class (class 0) for demonstration
shap_values_class_0 = shap_values[0]

# Get the feature importance (mean absolute SHAP value per feature)
feature_importance = np.mean(np.abs(shap_values_class_0), axis=0)

# Sort the features based on their importance
sorted_idx = np.argsort(feature_importance)[::-1]

# Display the most important features
print("Most important features based on SHAP values:")
for idx in sorted_idx:
    print(f"{X.columns[idx]}: {feature_importance[idx]:.4f}")

# Optionally, plot the SHAP values summary plot
# shap.summary_plot(shap_values_class_0, X)

