import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest


# ============================================================
# LOAD CLEANED DATA
# ============================================================

df = pd.read_csv(
    "cleaned_parking_data.csv"
)


# ============================================================
# SELECT SENSOR FEATURES
# ============================================================

features = [
    "Occupied_Spaces",
    "Vehicle_Entry",
    "Vehicle_Exit"
]


X = df[features]


# ============================================================
# CREATE ISOLATION FOREST MODEL
# ============================================================

model = IsolationForest(
    contamination=0.02,
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

df["Anomaly"] = model.fit_predict(
    X
)


# ============================================================
# CONVERT RESULT INTO READABLE STATUS
# ============================================================

df["Status"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})


# ============================================================
# COUNT RESULTS
# ============================================================

normal_count = (
    df["Status"] == "Normal"
).sum()


anomaly_count = (
    df["Status"] == "Anomaly"
).sum()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "================================"
)

print(
    "SMART PARKING ANOMALY DETECTION"
)

print(
    "================================"
)


print(
    "\nNormal readings:",
    normal_count
)


print(
    "Anomalous readings:",
    anomaly_count
)


print(
    "\nDetected Anomalies:"
)


anomalies = df[
    df["Status"] == "Anomaly"
]


print(
    anomalies[
        [
            "Time_Index",
            "Hour",
            "Occupied_Spaces",
            "Vehicle_Entry",
            "Vehicle_Exit",
            "Status"
        ]
    ].head(20)
)


# ============================================================
# SAVE ANOMALY RESULTS
# ============================================================

df.to_csv(
    "parking_anomaly_results.csv",
    index=False
)


# ============================================================
# SAVE TRAINED ANOMALY MODEL
# ============================================================

joblib.dump(
    model,
    "anomaly_model.pkl"
)


print(
    "\nAnomaly results saved successfully!"
)


print(
    "Anomaly model saved as anomaly_model.pkl"
)


print(
    "\n================================"
)

print(
    "ANOMALY MODEL READY FOR LIVE DATA"
)

print(
    "================================"
)