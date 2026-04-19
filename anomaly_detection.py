import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

# ── Generate realistic ESP32 DHT11 sensor data ──────────────────────────────
np.random.seed(42)
n = 500
timestamps = [datetime(2024, 1, 1) + timedelta(minutes=5 * i) for i in range(n)]

temperature = np.random.normal(loc=27, scale=1.5, size=n)
humidity = np.random.normal(loc=60, scale=5, size=n)

# Inject anomalies (sudden spikes — simulating sensor faults or env events)
anomaly_indices = [50, 120, 200, 310, 430]
temperature[anomaly_indices] += np.random.choice([10, -10, 12, -8, 11])
humidity[anomaly_indices] += np.random.choice([25, -20, 30, -25, 20])

df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature_c": np.round(temperature, 2),
    "humidity_pct": np.round(humidity, 2)
})

df.to_csv("sensor_data.csv", index=False)
print(f"Dataset created: {len(df)} readings, {len(anomaly_indices)} injected anomalies")

# ── Isolation Forest ─────────────────────────────────────────────────────────
features = df[["temperature_c", "humidity_pct"]]

model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(features)
df["anomaly_label"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

anomalies = df[df["anomaly"] == -1]
print(f"\nAnomalies detected: {len(anomalies)}")
print(anomalies[["timestamp", "temperature_c", "humidity_pct"]].to_string(index=False))

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle("ESP32 DHT11 Sensor — Anomaly Detection (Isolation Forest)", fontsize=14)

# Temperature
normal = df[df["anomaly"] == 1]
ax1.plot(normal["timestamp"], normal["temperature_c"], color="steelblue", linewidth=0.8, label="Normal")
ax1.scatter(anomalies["timestamp"], anomalies["temperature_c"], color="red", zorder=5, s=60, label="Anomaly")
ax1.set_ylabel("Temperature (°C)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Humidity
ax2.plot(normal["timestamp"], normal["humidity_pct"], color="green", linewidth=0.8, label="Normal")
ax2.scatter(anomalies["timestamp"], anomalies["humidity_pct"], color="red", zorder=5, s=60, label="Anomaly")
ax2.set_ylabel("Humidity (%)")
ax2.set_xlabel("Timestamp")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("anomaly_plot.png", dpi=150)
plt.show()
print("\nPlot saved as anomaly_plot.png")
