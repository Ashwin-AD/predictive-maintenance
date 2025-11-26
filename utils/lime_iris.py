import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

# Step 1: Load the Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 2: Train a RandomForest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Step 3: Initialize the LIME explainer
explainer = LimeTabularExplainer(
    training_data=X.values,  # The training data (as a numpy array)
    feature_names=X.columns,  # Feature names (columns of X)
    class_names=data.target_names,  # Class names (Iris species names)
    mode='classification'  # We are doing classification
)

# Step 4: Pick a sample to explain (let's explain the first sample)
i = 0
sample_to_explain = X.iloc[i].values

# Step 5: Generate the explanation using LIME
exp = explainer.explain_instance(
    data_row=sample_to_explain,  # The sample we want to explain
    predict_fn=model.predict_proba  # The model's prediction function
)

# Step 6: Display the explanation as a bar chart
# This will create a matplotlib figure and save/display it
fig = exp.as_pyplot_figure()

# Save the figure to a file
fig.savefig('lime_explanation.png')

# Alternatively, you can display it on screen
plt.show()

# Optionally, print the explanation as a list of feature contributions
print("Explanation (feature contributions):")
print(exp.as_list())

