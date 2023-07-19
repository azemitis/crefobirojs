import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('data.csv', header=0, names=["y", "v1", "v2", "v3", "v4", "v5", "v6", "v7"])

print("Predictor Summary:")
print(data.describe())

# Split the data into features (X) and dependent variable (y)
X = data.drop('y', axis=1)
y = data['y']

# Perform feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the base models on all predictors
model_rf_all = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
model_mlp_all = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)

model_rf_all.fit(X_train, y_train)
model_mlp_all.fit(X_train, y_train)

# Make predictions on the testing set using base models
pred_rf_all = model_rf_all.predict(X_test)
pred_mlp_all = model_mlp_all.predict(X_test)

# Stack the predictions from the base models
X_stacked_all = np.column_stack((pred_rf_all, pred_mlp_all))

# Create and train the meta-model (final model)
meta_model_all = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
meta_model_all.fit(X_stacked_all, y_test)

# Make predictions on the testing set
y_pred_all = meta_model_all.predict(X_stacked_all)

# Evaluate the model on all predictors
mse_all = mean_squared_error(y_test, y_pred_all)
r2_all = r2_score(y_test, y_pred_all)
print("Mean Squared Error without selecting predictors:", mse_all)
print("R^2 Score without selecting predictors:", r2_all)

# Solution with predictor selection
# Train a random forest model to calculate feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Calculate feature importance
feature_importance = rf.feature_importances_
print("Random Forest Feature Importance:")
for i, feature_name in enumerate(X.columns):
    print(f"{feature_name}: {feature_importance[i]}")

# Set a threshold for feature importance
threshold = 0.1

# Select predictors based on feature importance
selected_predictors = X.columns[feature_importance > threshold]

# Train the base models on selected predictors
model_rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
model_mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)

model_rf.fit(X_train[:, [X.columns.get_loc(predictor) for predictor in selected_predictors]], y_train)
model_mlp.fit(X_train[:, [X.columns.get_loc(predictor) for predictor in selected_predictors]], y_train)

# Make predictions on the testing set using base models
pred_rf = model_rf.predict(X_test[:, [X.columns.get_loc(predictor) for predictor in selected_predictors]])
pred_mlp = model_mlp.predict(X_test[:, [X.columns.get_loc(predictor) for predictor in selected_predictors]])

# Stack the predictions from the base models as new features
X_stacked = np.column_stack((pred_rf, pred_mlp))

# Create and train the meta-model (final model)
meta_model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, random_state=42)
meta_model.fit(X_stacked, y_test)

# Make predictions on the testing set using the meta-model
y_pred = meta_model.predict(X_stacked)

# Evaluate the model with selected predictors
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error with selected predictors:", mse)
print("R^2 Score with selected predictors:", r2)
