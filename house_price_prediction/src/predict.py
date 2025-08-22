import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("./models/house_model.pkl")
scaler = joblib.load("./models/scaler.pkl")

# Example input (must follow dataset feature order)
data = {
    "MedInc": [8.3252],      # Median Income in block
    "HouseAge": [41],        # House Age
    "AveRooms": [6.9841],    # Average Rooms
    "AveBedrms": [1.0238],   # Average Bedrooms
    "Population": [322],     # Population
    "AveOccup": [2.5556],    # Average Occupancy
    "Latitude": [37.88],     # Latitude
    "Longitude": [-122.23]   # Longitude
}

# Convert to DataFrame
example_house = pd.DataFrame(data)

# Scale input
example_house_scaled = scaler.transform(example_house)

# Predict
prediction = model.predict(example_house_scaled)
print("Predicted House Price:", prediction[0])
