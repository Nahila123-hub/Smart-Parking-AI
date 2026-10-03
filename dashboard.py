import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Great+Vibes&family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, #243b55 0%, transparent 35%),
        radial-gradient(circle at bottom right, #1d4e89 0%, transparent 35%),
        linear-gradient(135deg, #07111f, #0b1c30, #102b46);
    color: #f4f8fc;
}

.main-title {
    font-family: 'Great Vibes', cursive;
    font-size: 65px;
    text-align: center;
    margin-bottom: 0px;
    color: #dceeff;
    text-shadow: 0px 3px 15px rgba(130, 200, 255, 0.25);
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #b8d4ed;
    margin-bottom: 20px;
}

.hero-badge {
    text-align: center;
    margin: auto;
    width: fit-content;
    padding: 8px 22px;
    border-radius: 30px;
    background: rgba(76, 175, 240, 0.12);
    border: 1px solid rgba(150, 210, 255, 0.25);
    color: #bfe4ff;
    font-size: 14px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 26px;
    font-weight: 600;
    color: #d8ecff;
    margin-top: 25px;
    margin-bottom: 15px;
}

.card {
    background: rgba(255, 255, 255, 0.075);
    border: 1px solid rgba(190, 225, 255, 0.13);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.18);
}

.metric-card {
    background: rgba(255, 255, 255, 0.075);
    border: 1px solid rgba(190, 225, 255, 0.13);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    min-height: 125px;
}

.metric-number {
    font-size: 32px;
    font-weight: 700;
    color: #d9efff;
}

.metric-label {
    color: #a9c5dc;
    font-size: 14px;
}

.live-badge {
    background: rgba(70, 190, 120, 0.15);
    border: 1px solid rgba(100, 230, 150, 0.3);
    color: #9ff0bc;
    padding: 7px 15px;
    border-radius: 20px;
    display: inline-block;
    font-size: 13px;
}

.normal-badge {
    background: rgba(70, 190, 120, 0.15);
    border: 1px solid rgba(100, 230, 150, 0.3);
    color: #9ff0bc;
    padding: 8px 15px;
    border-radius: 20px;
    display: inline-block;
}

.anomaly-badge {
    background: rgba(240, 80, 80, 0.15);
    border: 1px solid rgba(255, 110, 110, 0.35);
    color: #ff9c9c;
    padding: 8px 15px;
    border-radius: 20px;
    display: inline-block;
}

.sensor-box {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(180,220,255,0.12);
    border-radius: 14px;
    padding: 15px;
    text-align: center;
}

.sensor-value {
    font-size: 25px;
    font-weight: 600;
    color: #dceeff;
}

.sensor-name {
    color: #a9c5dc;
    font-size: 13px;
}

.slot {
    width: 55px;
    height: 55px;
    border-radius: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 5px;
    font-weight: 600;
    font-size: 13px;
}

.slot-free {
    background: rgba(65, 190, 115, 0.20);
    border: 1px solid rgba(100, 240, 150, 0.45);
    color: #9ff0bc;
}

.slot-occupied {
    background: rgba(240, 75, 75, 0.20);
    border: 1px solid rgba(255, 100, 100, 0.45);
    color: #ffaaaa;
}

.footer {
    text-align: center;
    color: #7895ad;
    font-size: 13px;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

try:
    df = pd.read_csv("parking_anomaly_results.csv")
    prediction_model = joblib.load("prediction_model.pkl")
    anomaly_model = joblib.load("anomaly_model.pkl")

except Exception as e:
    st.error("Required files could not be loaded.")
    st.error(str(e))
    st.stop()


# ============================================================
# BASIC INFORMATION
# ============================================================

TOTAL_SPACES = 50

slot_columns = []

for number in range(1, TOTAL_SPACES + 1):
    slot_columns.append(f"Slot_{number}")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

X = df[
    [
        "Hour",
        "Occupied_Spaces",
        "Vehicle_Entry",
        "Vehicle_Exit"
    ]
]

y = df["Future_Available_Spaces"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


historical_predictions = prediction_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    historical_predictions
)

r2 = r2_score(
    y_test,
    historical_predictions
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">Smart Parking AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Smart Parking Availability Prediction & Anomaly Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-badge">🚗 IoT Sensors • 🤖 Machine Learning • 📊 Real-Time Analytics</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "rng" not in st.session_state:
    st.session_state.rng = np.random.default_rng(42)


if "live_slots" not in st.session_state:

    slots = np.zeros(TOTAL_SPACES, dtype=int)

    initial_occupied = st.session_state.rng.choice(
        TOTAL_SPACES,
        size=20,
        replace=False
    )

    slots[initial_occupied] = 1

    st.session_state.live_slots = slots


if "simulation_minutes" not in st.session_state:
    st.session_state.simulation_minutes = 8 * 60


if "live_history" not in st.session_state:
    st.session_state.live_history = []


if "live_tick" not in st.session_state:
    st.session_state.live_tick = 0


if "last_entry" not in st.session_state:
    st.session_state.last_entry = 0


if "last_exit" not in st.session_state:
    st.session_state.last_exit = 0


if "last_anomaly" not in st.session_state:
    st.session_state.last_anomaly = "Normal"


# ============================================================
# RESET LIVE SIMULATION
# ============================================================

reset_column, status_column = st.columns([1, 3])

with reset_column:

    if st.button("🔄 Reset Live Simulation"):

        slots = np.zeros(TOTAL_SPACES, dtype=int)

        initial_occupied = st.session_state.rng.choice(
            TOTAL_SPACES,
            size=20,
            replace=False
        )

        slots[initial_occupied] = 1

        st.session_state.live_slots = slots
        st.session_state.simulation_minutes = 8 * 60
        st.session_state.live_history = []
        st.session_state.live_tick = 0
        st.session_state.last_entry = 0
        st.session_state.last_exit = 0
        st.session_state.last_anomaly = "Normal"

        st.rerun()


with status_column:

    st.markdown(
        '<span class="live-badge">● LIVE SIMULATION ACTIVE</span>',
        unsafe_allow_html=True
    )


# ============================================================
# LIVE SIMULATION FUNCTION
# ============================================================

def update_live_simulation():

    rng = st.session_state.rng

    slots = st.session_state.live_slots

    occupied_before = int(np.sum(slots))

    current_hour = (
        st.session_state.simulation_minutes // 60
    ) % 24


    # --------------------------------------------------------
    # TRAFFIC PATTERN
    # --------------------------------------------------------

    if 8 <= current_hour <= 10:

        entry_rate = 5
        exit_rate = 2

    elif 17 <= current_hour <= 20:

        entry_rate = 6
        exit_rate = 3

    elif current_hour >= 22 or current_hour <= 6:

        entry_rate = 1
        exit_rate = 4

    else:

        entry_rate = 3
        exit_rate = 3


    # --------------------------------------------------------
    # KEEP PARKING OCCUPANCY REALISTIC
    # --------------------------------------------------------

    if occupied_before <= 12:

        entry_rate += 4
        exit_rate = max(1, exit_rate - 1)

    elif occupied_before >= 38:

        exit_rate += 4
        entry_rate = max(1, entry_rate - 1)


    # --------------------------------------------------------
    # VEHICLE EXIT
    # --------------------------------------------------------

    requested_exit = int(
        rng.poisson(exit_rate)
    )

    occupied_indices = np.where(slots == 1)[0]

    actual_exit = min(
        requested_exit,
        len(occupied_indices)
    )


    if actual_exit > 0:

        exiting_slots = rng.choice(
            occupied_indices,
            size=actual_exit,
            replace=False
        )

        slots[exiting_slots] = 0


    # --------------------------------------------------------
    # VEHICLE ENTRY
    # --------------------------------------------------------

    requested_entry = int(
        rng.poisson(entry_rate)
    )

    available_indices = np.where(slots == 0)[0]

    actual_entry = min(
        requested_entry,
        len(available_indices)
    )


    if actual_entry > 0:

        entering_slots = rng.choice(
            available_indices,
            size=actual_entry,
            replace=False
        )

        slots[entering_slots] = 1


    # --------------------------------------------------------
    # LIVE SENSOR VALUES
    # --------------------------------------------------------

    sensor_entry = actual_entry
    sensor_exit = actual_exit


    # --------------------------------------------------------
    # OCCASIONAL SENSOR ANOMALY
    # --------------------------------------------------------

    if rng.random() < 0.04:

        if rng.random() < 0.5:

            sensor_entry = min(
                25,
                actual_entry + int(rng.integers(10, 18))
            )

        else:

            sensor_exit = min(
                25,
                actual_exit + int(rng.integers(10, 18))
            )


    # --------------------------------------------------------
    # CURRENT PARKING STATE
    # --------------------------------------------------------

    occupied = int(np.sum(slots))

    available = TOTAL_SPACES - occupied


    # --------------------------------------------------------
    # ADVANCE SIMULATED TIME
    # --------------------------------------------------------

    st.session_state.simulation_minutes += 5

    st.session_state.live_tick += 1

    simulated_hour = (
        st.session_state.simulation_minutes // 60
    ) % 24


    # --------------------------------------------------------
    # AI PREDICTION
    # --------------------------------------------------------

    prediction_input = pd.DataFrame({

        "Hour": [simulated_hour],

        "Occupied_Spaces": [occupied],

        "Vehicle_Entry": [sensor_entry],

        "Vehicle_Exit": [sensor_exit]

    })


    predicted_available = prediction_model.predict(
        prediction_input
    )[0]


    predicted_available = int(
        round(predicted_available)
    )


    predicted_available = max(
        0,
        min(TOTAL_SPACES, predicted_available)
    )


    # --------------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------------

    anomaly_input = pd.DataFrame({

        "Occupied_Spaces": [occupied],

        "Vehicle_Entry": [sensor_entry],

        "Vehicle_Exit": [sensor_exit]

    })


    anomaly_prediction = anomaly_model.predict(
        anomaly_input
    )[0]


    if anomaly_prediction == -1:

        anomaly_status = "Anomaly"

    else:

        anomaly_status = "Normal"


    # --------------------------------------------------------
    # STORE CURRENT STATE
    # --------------------------------------------------------

    st.session_state.live_slots = slots

    st.session_state.last_entry = sensor_entry

    st.session_state.last_exit = sensor_exit

    st.session_state.last_anomaly = anomaly_status


    # --------------------------------------------------------
    # LIVE HISTORY
    # --------------------------------------------------------

    current_time = (
        f"{simulated_hour:02d}:"
        f"{st.session_state.simulation_minutes % 60:02d}"
    )


    record = {

        "Tick":
            st.session_state.live_tick,

        "Time":
            current_time,

        "Hour":
            simulated_hour,

        "Occupied":
            occupied,

        "Available":
            available,

        "Entry":
            sensor_entry,

        "Exit":
            sensor_exit,

        "Predicted_Available":
            predicted_available,

        "Anomaly":
            anomaly_status

    }


    st.session_state.live_history.append(
        record
    )


    # Keep only latest 60 readings

    if len(st.session_state.live_history) > 60:

        st.session_state.live_history = (
            st.session_state.live_history[-60:]
        )


# ============================================================
# REAL-TIME DASHBOARD FRAGMENT
# ============================================================

@st.fragment(run_every="3s")
def live_dashboard():

    # --------------------------------------------------------
    # UPDATE LIVE DATA
    # --------------------------------------------------------

    update_live_simulation()


    slots = st.session_state.live_slots

    occupied = int(np.sum(slots))

    available = TOTAL_SPACES - occupied

    entry = st.session_state.last_entry

    exit_count = st.session_state.last_exit

    anomaly_status = st.session_state.last_anomaly


    current_hour = (
        st.session_state.simulation_minutes // 60
    ) % 24


    current_minute = (
        st.session_state.simulation_minutes % 60
    )


    # ========================================================
    # LIVE OVERVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">⚡ Real-Time Parking Overview</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{occupied}</div>
                <div class="metric-label">Occupied Spaces</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{available}</div>
                <div class="metric-label">Available Spaces</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{entry}</div>
                <div class="metric-label">Vehicle Entries</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{exit_count}</div>
                <div class="metric-label">Vehicle Exits</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # CURRENT SIMULATED TIME
    # ========================================================

    st.markdown(
        f"""
        <div class="card">
            🕒 <b>Simulated Time:</b>
            {current_hour:02d}:{current_minute:02d}
            &nbsp;&nbsp; | &nbsp;&nbsp;
            🔄 <b>Simulation Tick:</b>
            {st.session_state.live_tick}
            &nbsp;&nbsp; | &nbsp;&nbsp;
            ⏱️ <b>Update Interval:</b>
            3 seconds
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PARKING AVAILABILITY
    # ========================================================

    st.markdown(
        '<div class="section-title">🅿️ Current Parking Availability</div>',
        unsafe_allow_html=True
    )


    occupancy_percentage = (
        occupied / TOTAL_SPACES
    ) * 100


    st.progress(
        occupancy_percentage / 100
    )


    st.write(
        f"**{occupied} / {TOTAL_SPACES} spaces occupied "
        f"({occupancy_percentage:.1f}%)**"
    )


    # ========================================================
    # PARKING SLOT MAP
    # ========================================================

    st.markdown(
        '<div class="section-title">🚘 Live Parking Slot Map</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "🟢 Green = Available    🔴 Red = Occupied"
    )


    for row_start in range(0, TOTAL_SPACES, 10):

        columns = st.columns(10)


        for position in range(10):

            slot_index = row_start + position

            slot_number = slot_index + 1

            slot_state = slots[slot_index]


            with columns[position]:

                if slot_state == 1:

                    st.markdown(
                        f"""
                        <div class="slot slot-occupied">
                            P{slot_number}<br>🔴
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="slot slot-free">
                            P{slot_number}<br>🟢
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    # ========================================================
    # LIVE SENSOR READINGS
    # ========================================================

    st.markdown(
        '<div class="section-title">📡 Live IoT Sensor Readings</div>',
        unsafe_allow_html=True
    )


    sensor1, sensor2, sensor3, sensor4 = st.columns(4)


    with sensor1:

        st.markdown(
            f"""
            <div class="sensor-box">
                <div class="sensor-value">{occupied}</div>
                <div class="sensor-name">
                    Occupancy Sensor
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with sensor2:

        st.markdown(
            f"""
            <div class="sensor-box">
                <div class="sensor-value">{entry}</div>
                <div class="sensor-name">
                    Vehicle Entry Sensor
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with sensor3:

        st.markdown(
            f"""
            <div class="sensor-box">
                <div class="sensor-value">{exit_count}</div>
                <div class="sensor-name">
                    Vehicle Exit Sensor
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with sensor4:

        st.markdown(
            f"""
            <div class="sensor-box">
                <div class="sensor-value">{available}</div>
                <div class="sensor-name">
                    Available Spaces
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # AI PREDICTION + ANOMALY
    # ========================================================

    history_df = pd.DataFrame(
        st.session_state.live_history
    )


    latest_live = history_df.iloc[-1]


    predicted_available = int(
        latest_live["Predicted_Available"]
    )


    st.markdown(
        '<div class="section-title">🤖 Live AI Analysis</div>',
        unsafe_allow_html=True
    )


    ai_col1, ai_col2, ai_col3 = st.columns(3)


    with ai_col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {predicted_available}
                </div>
                <div class="metric-label">
                    AI Predicted Future Availability
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with ai_col2:

        if anomaly_status == "Normal":

            st.markdown(
                """
                <div class="metric-card">
                    <div class="normal-badge">
                        ✓ NORMAL
                    </div>
                    <br><br>
                    <div class="metric-label">
                        Isolation Forest Status
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="metric-card">
                    <div class="anomaly-badge">
                        ⚠ ANOMALY
                    </div>
                    <br><br>
                    <div class="metric-label">
                        Isolation Forest Status
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    with ai_col3:

        prediction_difference = (
            predicted_available - available
        )


        if prediction_difference > 0:

            direction = "📈 Expected Increase"

        elif prediction_difference < 0:

            direction = "📉 Expected Decrease"

        else:

            direction = "➡ Stable"


        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {abs(prediction_difference)}
                </div>
                <div class="metric-label">
                    {direction}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # LIVE OCCUPANCY GRAPH
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Live Parking Occupancy</div>',
        unsafe_allow_html=True
    )


    if len(history_df) >= 2:

        fig1, ax1 = plt.subplots(
            figsize=(12, 4)
        )

        ax1.plot(
            history_df["Tick"],
            history_df["Occupied"],
            label="Occupied Spaces",
            linewidth=2
        )

        ax1.plot(
            history_df["Tick"],
            history_df["Available"],
            label="Available Spaces",
            linewidth=2
        )

        ax1.set_xlabel("Live Simulation Tick")

        ax1.set_ylabel("Number of Spaces")

        ax1.set_title(
            "Real-Time Parking Occupancy"
        )

        ax1.legend()

        ax1.grid(True, alpha=0.25)

        st.pyplot(fig1)

        plt.close(fig1)


    # ========================================================
    # LIVE AI PREDICTION GRAPH
    # ========================================================

    st.markdown(
        '<div class="section-title">🤖 Live AI Availability Prediction</div>',
        unsafe_allow_html=True
    )


    if len(history_df) >= 2:

        fig2, ax2 = plt.subplots(
            figsize=(12, 4)
        )

        ax2.plot(
            history_df["Tick"],
            history_df["Available"],
            label="Actual Available Spaces",
            linewidth=2
        )

        ax2.plot(
            history_df["Tick"],
            history_df["Predicted_Available"],
            label="AI Predicted Availability",
            linewidth=2
        )

        ax2.set_xlabel(
            "Live Simulation Tick"
        )

        ax2.set_ylabel(
            "Available Spaces"
        )

        ax2.set_title(
            "Real-Time Actual vs AI Predicted Availability"
        )

        ax2.legend()

        ax2.grid(True, alpha=0.25)

        st.pyplot(fig2)

        plt.close(fig2)


    # ========================================================
    # LIVE SENSOR HISTORY
    # ========================================================

    st.markdown(
        '<div class="section-title">📡 Live Sensor Data Stream</div>',
        unsafe_allow_html=True
    )


    display_columns = [
        "Tick",
        "Time",
        "Hour",
        "Occupied",
        "Available",
        "Entry",
        "Exit",
        "Predicted_Available",
        "Anomaly"
    ]


    st.dataframe(
        history_df[display_columns].tail(10).iloc[::-1],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RUN LIVE DASHBOARD
# ============================================================

live_dashboard()


# ============================================================
# HISTORICAL ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Historical Parking Analytics</div>',
    unsafe_allow_html=True
)


historical_col1, historical_col2 = st.columns(2)


with historical_col1:

    fig3, ax3 = plt.subplots(
        figsize=(8, 4)
    )

    ax3.plot(
        df["Time_Index"],
        df["Occupied_Spaces"],
        linewidth=1.5
    )

    ax3.set_title(
        "Historical Parking Occupancy"
    )

    ax3.set_xlabel(
        "Time Index"
    )

    ax3.set_ylabel(
        "Occupied Spaces"
    )

    ax3.grid(True, alpha=0.25)

    st.pyplot(fig3)

    plt.close(fig3)


with historical_col2:

    fig4, ax4 = plt.subplots(
        figsize=(8, 4)
    )

    ax4.plot(
        df["Time_Index"],
        df["Available_Spaces"],
        linewidth=1.5
    )

    ax4.set_title(
        "Historical Parking Availability"
    )

    ax4.set_xlabel(
        "Time Index"
    )

    ax4.set_ylabel(
        "Available Spaces"
    )

    ax4.grid(True, alpha=0.25)

    st.pyplot(fig4)

    plt.close(fig4)


# ============================================================
# HISTORICAL ANOMALY MONITORING
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Historical Anomaly Monitoring</div>',
    unsafe_allow_html=True
)


normal_count = (
    df["Status"] == "Normal"
).sum()


anomaly_count = (
    df["Status"] == "Anomaly"
).sum()


anom_col1, anom_col2 = st.columns(2)


with anom_col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {normal_count}
            </div>
            <div class="metric-label">
                Historical Normal Readings
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with anom_col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {anomaly_count}
            </div>
            <div class="metric-label">
                Historical Anomalies
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    "### Detected Historical Anomalies"
)


historical_anomalies = df[
    df["Status"] == "Anomaly"
]


st.dataframe(
    historical_anomalies[
        [
            "Time_Index",
            "Hour",
            "Occupied_Spaces",
            "Vehicle_Entry",
            "Vehicle_Exit",
            "Status"
        ]
    ].head(20),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Prediction Model Performance</div>',
    unsafe_allow_html=True
)


performance_col1, performance_col2 = st.columns(2)


with performance_col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {mae:.2f}
            </div>
            <div class="metric-label">
                Mean Absolute Error
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with performance_col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {r2:.2f}
            </div>
            <div class="metric-label">
                R² Score
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

st.markdown(
    '<div class="section-title">📈 Historical Actual vs Predicted Availability</div>',
    unsafe_allow_html=True
)


fig5, ax5 = plt.subplots(
    figsize=(12, 5)
)


ax5.plot(
    range(len(y_test)),
    y_test.values,
    label="Actual",
    linewidth=2
)


ax5.plot(
    range(len(historical_predictions)),
    historical_predictions,
    label="Predicted",
    linewidth=2
)


ax5.set_title(
    "Historical Model Evaluation"
)


ax5.set_xlabel(
    "Test Samples"
)


ax5.set_ylabel(
    "Available Spaces"
)


ax5.legend()

ax5.grid(True, alpha=0.25)

st.pyplot(fig5)

plt.close(fig5)


# ============================================================
# AI SYSTEM INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🧠 AI System Information</div>',
    unsafe_allow_html=True
)


info_col1, info_col2 = st.columns(2)


with info_col1:

    st.markdown(
        """
        <div class="card">

        <b>Prediction Model</b><br><br>

        Linear Regression predicts the number of
        parking spaces expected to be available
        in the next interval.

        <br><br>

        <b>Input Sensors</b>

        <ul>
            <li>Hour</li>
            <li>Occupied Spaces</li>
            <li>Vehicle Entry</li>
            <li>Vehicle Exit</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )


with info_col2:

    st.markdown(
        """
        <div class="card">

        <b>Anomaly Detection</b><br><br>

        Isolation Forest identifies unusual
        combinations of parking occupancy,
        vehicle entries and vehicle exits.

        <br><br>

        <b>Parking Sensors</b>

        <ul>
            <li>50 individual slot sensors</li>
            <li>1 = Occupied</li>
            <li>0 = Available</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown(
    '<div class="section-title">🟢 System Status</div>',
    unsafe_allow_html=True
)


status1, status2, status3, status4 = st.columns(4)


with status1:

    st.success(
        "✓ IoT Simulation Online"
    )


with status2:

    st.success(
        "✓ Prediction Model Active"
    )


with status3:

    st.success(
        "✓ Anomaly Detection Active"
    )


with status4:

    st.success(
        "✓ Live Dashboard Active"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    Smart Parking AI • AI for IoT Capstone Project<br>

    Simulated IoT Sensors + Machine Learning + Real-Time Dashboard

    </div>
    """,
    unsafe_allow_html=True
)