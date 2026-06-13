import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load("models/linear_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Enter House Details")

medinc = float(input("Median Income: "))
houseage = float(input("House Age: "))
averooms = float(input("Average Rooms: "))
avebedrms = float(input("Average Bedrooms: "))
population = float(input("Population: "))
aveoccup = float(input("Average Occupancy: "))
latitude = float(input("Latitude: "))
longitude = float(input("Longitude: "))

if latitude < 32.54 or latitude > 41.95:
    print("Invalid latitude for California dataset so please enter the values between 32.54 to 41.95")
    exit()

if longitude < -124.35 or longitude > -114.31:
    print("Invalid longitude for California dataset so please enter the values between -124.35-to-114.31")
    exit()

data = pd.DataFrame([[
    medinc,
    houseage,
    averooms,
    avebedrms,
    population,
    aveoccup,
    latitude,
    longitude
]], columns=[
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
])

# Scale input
data_scaled = scaler.transform(data)

# Predict
prediction = model.predict(data_scaled)

print("\nPredicted House Price:")
print(f"${prediction[0]:.2f} (hundreds of thousands of dollars)")
