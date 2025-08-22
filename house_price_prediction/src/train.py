import os
import joblib
from preprocess import load_and_preprocess
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Ensure models directory exists
os.makedirs("./models", exist_ok=True)

# Save model & scaler
joblib.dump(model, "./models/house_model.pkl")
joblib.dump(scaler, "./models/scaler.pkl")
