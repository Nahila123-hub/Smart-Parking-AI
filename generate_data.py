import numpy as np
import pandas as pd


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(42)


# ============================================================
# PARKING CONFIGURATION
# ============================================================

total_spaces = 50

number_of_records = 1000


# ============================================================
# CREATE 50 SIMULATED PARKING SLOT SENSORS
# ============================================================

slot_names = []

for slot_number in range(1, total_spaces + 1):

    slot_names.append(
        f"Slot_{slot_number}"
    )


# ============================================================
# INITIAL PARKING CONDITION
# ============================================================

# 20 vehicles are initially parked.

occupied = 20


# 0 = Available
# 1 = Occupied

slot_status = np.zeros(
    total_spaces,
    dtype=int
)


# Randomly select 20 slots as occupied

initial_occupied_slots = np.random.choice(
    total_spaces,
    size=occupied,
    replace=False
)


slot_status[
    initial_occupied_slots
] = 1


# ============================================================
# STORE DATA
# ============================================================

data = []


# ============================================================
# SIMULATE PARKING SYSTEM
# ============================================================

for i in range(number_of_records):

    # Simulated hour
    hour = (
        (i // 10) % 24
    )


    # --------------------------------------------------------
    # TRAFFIC PATTERNS
    # --------------------------------------------------------

    if 8 <= hour <= 10:

        entry_rate = 7
        exit_rate = 3

    elif 17 <= hour <= 20:

        entry_rate = 8
        exit_rate = 4

    elif 22 <= hour or hour <= 6:

        entry_rate = 2
        exit_rate = 5

    else:

        entry_rate = 4
        exit_rate = 4


    # --------------------------------------------------------
    # SIMULATED ENTRY SENSOR
    # --------------------------------------------------------

    vehicle_entry = max(
        0,
        int(
            np.random.normal(
                entry_rate,
                2
            )
        )
    )


    # --------------------------------------------------------
    # SIMULATED EXIT SENSOR
    # --------------------------------------------------------

    vehicle_exit = max(
        0,
        int(
            np.random.normal(
                exit_rate,
                2
            )
        )
    )


    # ========================================================
    # FIND CURRENT AVAILABLE AND OCCUPIED SLOTS
    # ========================================================

    available_slot_indices = np.where(
        slot_status == 0
    )[0]


    occupied_slot_indices = np.where(
        slot_status == 1
    )[0]


    # ========================================================
    # LIMIT ENTRIES AND EXITS
    # ========================================================

    # Cannot enter if there are no free spaces.

    actual_entries = min(
        vehicle_entry,
        len(available_slot_indices)
    )


    # Cannot exit more vehicles than are parked.

    actual_exits = min(
        vehicle_exit,
        len(occupied_slot_indices)
    )


    # ========================================================
    # VEHICLE EXITS
    # ========================================================

    if actual_exits > 0:

        exiting_slots = np.random.choice(
            occupied_slot_indices,
            size=actual_exits,
            replace=False
        )

        slot_status[
            exiting_slots
        ] = 0


    # ========================================================
    # VEHICLE ENTRIES
    # ========================================================

    # Find free slots again after vehicles exit.

    available_slot_indices = np.where(
        slot_status == 0
    )[0]


    actual_entries = min(
        actual_entries,
        len(available_slot_indices)
    )


    if actual_entries > 0:

        entering_slots = np.random.choice(
            available_slot_indices,
            size=actual_entries,
            replace=False
        )

        slot_status[
            entering_slots
        ] = 1


    # ========================================================
    # CALCULATE OCCUPANCY FROM INDIVIDUAL SLOT SENSORS
    # ========================================================

    occupied = int(
        np.sum(slot_status)
    )


    available_spaces = (
        total_spaces - occupied
    )


    # ========================================================
    # SAVE CURRENT SENSOR READING
    # ========================================================

    row = [
        i,
        hour,
        occupied,
        actual_entries,
        actual_exits,
        available_spaces
    ]


    # Add all 50 parking slot sensor values

    row.extend(
        slot_status.tolist()
    )


    data.append(row)


# ============================================================
# CREATE COLUMN NAMES
# ============================================================

columns = [
    "Time_Index",
    "Hour",
    "Occupied_Spaces",
    "Vehicle_Entry",
    "Vehicle_Exit",
    "Available_Spaces"
]


columns.extend(
    slot_names
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    data,
    columns=columns
)


# ============================================================
# CREATE FUTURE AVAILABILITY TARGET
# ============================================================

df["Future_Available_Spaces"] = (
    df["Available_Spaces"].shift(-1)
)


# Remove final row because it has
# no future availability value.

df = df.dropna()


# ============================================================
# ADD SIMULATED SENSOR ANOMALIES
# ============================================================

# These represent faulty entry/exit sensor readings.
# The actual slot occupancy remains unchanged.

anomaly_indices = [
    150,
    350,
    600,
    850
]


for index in anomaly_indices:

    if index < len(df):

        # Abnormally high entry reading

        df.loc[
            index,
            "Vehicle_Entry"
        ] = 25


        # Abnormally high exit reading

        df.loc[
            index,
            "Vehicle_Exit"
        ] = 20


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    "parking_data.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "New Smart Parking dataset generated successfully!"
)


print(
    "\nDataset preview:"
)

print(
    df.head(10)
)


print(
    "\nDataset shape:"
)

print(
    df.shape
)


print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# VERIFY SLOT SENSOR LOGIC
# ============================================================

print(
    "\n============================================"
)

print(
    "SLOT SENSOR VERIFICATION"
)

print(
    "============================================"
)


first_row = df.iloc[0]


calculated_occupied = int(
    df.loc[
        df.index[0],
        slot_names
    ].sum()
)


calculated_available = (
    total_spaces
    - calculated_occupied
)


print(
    "\nOccupied Spaces:",
    int(first_row["Occupied_Spaces"])
)


print(
    "Occupied from Slot Sensors:",
    calculated_occupied
)


print(
    "Available Spaces:",
    int(first_row["Available_Spaces"])
)


print(
    "Available from Slot Sensors:",
    calculated_available
)


if (
    calculated_occupied
    == int(first_row["Occupied_Spaces"])
):

    print(
        "\n✓ Slot sensor occupancy is consistent."
    )

else:

    print(
        "\n✗ Slot sensor occupancy mismatch."
    )


print(
    "\nNumber of parking slot sensors:",
    total_spaces
)


print(
    "Dataset generation completed successfully!"
)