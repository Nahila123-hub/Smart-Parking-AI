import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load cleaned data
df = pd.read_csv("cleaned_parking_data.csv")

# Input features
X = df[
    [
        "Hour",
        "Occupied_Spaces",
        "Vehicle_Entry",
        "Vehicle_Exit"
    ]
]

# Target = FUTURE availability
y = df["Future_Available_Spaces"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n================================")
print("SMART PARKING PREDICTION MODEL")
print("================================")

print("\nMean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Save model
joblib.dump(model, "prediction_model.pkl")

print("\nModel saved as prediction_model.pkl")

# Show sample predictions
results = pd.DataFrame({
    "Actual_Future_Availability": y_test.values,
    "Predicted_Future_Availability": y_pred.round(2)
})

print("\nSample Predictions:")
print(results.head(10))

# Plot
plt.figure(figsize=(10, 5))

plt.plot(
    range(len(y_test)),
    y_test.values,
    label="Actual"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    label="Predicted"
)

plt.title("Future Parking Availability: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Available Spaces")

plt.legend()
plt.grid()

plt.show()