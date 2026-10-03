import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("parking_data.csv")

plt.figure(figsize=(10, 5))

plt.plot(
    df["Time_Index"],
    df["Occupied_Spaces"]
)

plt.title("Parking Occupancy Over Time")
plt.xlabel("Time")
plt.ylabel("Occupied Spaces")

plt.grid()
plt.show()