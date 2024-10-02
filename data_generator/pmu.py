import numpy as np
from datetime import datetime, timedelta
import random
import pandas as pd
import matplotlib.pyplot as plt


class VirtualPMU:
    def __init__(self, sample_rate=60, event_rate=0.02, error_rate=0.03):
        self.sample_rate = sample_rate
        self.error_rate = error_rate  # Percentage of data points to be null due to communication errors
        self.event_rate = event_rate

    def generate_complex_pmu_data(self, duration=1):
        t = np.arange(0, duration, 1 / self.sample_rate)

        # Nominal frequency set to 60Hz with small variations
        # frequency = 60 + np.random.normal(0, 0.01) * np.sin(2 * np.pi * 0.1 * t)
        # voltage = 1 + np.sin(2 * np.pi * 0.2 * t) * np.random.normal(0, 0.015)
        # current = 0.95 + np.sin(2 * np.pi * 0.3 * t) * random.uniform(0.01, 0.015)
        # phase_angle = np.sin(2 * np.pi * 0.4 * t) + random.uniform(0.01, 0.015)

        frequency = 60 + np.random.normal(0, 0.01, t.shape[0])
        voltage = 1 + np.random.normal(0, 0.015, t.shape[0])
        current = 0.95 + np.random.normal(0, 0.015, t.shape[0])
        phase_angle = 0 + np.random.normal(0, 0.015, t.shape[0])

        base_timestamp = datetime.now()
        timestamps = [base_timestamp + timedelta(seconds=i) for i in range(len(t))]

        anomaly_log = []
        # Simulate specific complex anomalies:
        if random.random() < self.event_rate:
            # 1. Frequency Deviations
            st = int(frequency.shape[0] * random.random())
            frequency[st : st + 10] -= 0.2  # Frequency drops
            for i in range(10):
                anomaly_log.append((timestamps[st + i], "frequency", "drops"))

        if random.random() < self.event_rate:
            # 2. Frequency spike
            st = int(frequency.shape[0] * random.random())
            frequency[st : st + 10] += 0.2
            for i in range(10):
                anomaly_log.append((timestamps[st + i], "frequency", "spike"))

        if random.random() < self.event_rate:
            # 3. Phase Angle Shift
            st = int(frequency.shape[0] * random.random())
            phase_angle[st : st + 10] += 0.3  # Phase angle shift
            for i in range(10):
                anomaly_log.append((timestamps[st + i], "phase_angle", "shift"))

        if random.random() < self.event_rate:
            # 4. Voltage Sag
            st = int(frequency.shape[0] * random.random())
            voltage[st : st + 10] -= 0.3  # Voltage sag

            for i in range(10):
                anomaly_log.append((timestamps[st + i], "voltage", "sag"))

        if random.random() < self.event_rate:
            # 5. Voltage Surge
            st = int(frequency.shape[0] * random.random())
            voltage[st : st + 10] += 0.3  # Voltage surge
            for i in range(10):
                anomaly_log.append((timestamps[st + i], "voltage", "surge"))

        if random.random() < self.event_rate:
            st = int(frequency.shape[0] * random.random())
            # 6. Oscillatory Behavior (Simulate damped oscillations in voltage)
            oscillation = (
                0.05
                * np.sin(2 * np.pi * 5 * t[st : st + 20])
                * np.exp(-0.1 * t[st : st + 20])
            )
            voltage[st : st + 20] += oscillation
            for i in range(20):
                anomaly_log.append(
                    (timestamps[st + i], "voltage", "voltage oscilation")
                )

        if random.random() < self.event_rate:
            # 7. Frequency Oscillation
            st = int(frequency.shape[0] * random.random())
            frequency[st : st + 20] += 0.5 * np.sin(
                2 * np.pi * 10 * t[st : st + 20]
            )  # High-frequency oscillations
            for i in range(20):
                anomaly_log.append((timestamps[st + i], "frequency", "high oscilation"))

        if random.random() < self.event_rate:
            # 8. Frequency Oscillation
            st = int(frequency.shape[0] * random.random())
            frequency[st : st + 20] += 0.1 * np.sin(
                2 * np.pi * 0.5 * t[st : st + 20]
            )  # low-frequency oscillations
            for i in range(20):
                anomaly_log.append((timestamps[st + i], "frequency", "low oscilation"))

        # Introduce random null values (representing communication errors)
        for i in range(len(frequency)):
            if random.random() < self.error_rate:
                frequency[i] = np.nan
            if random.random() < self.error_rate:
                voltage[i] = np.nan
            if random.random() < self.error_rate:
                current[i] = np.nan
            if random.random() < self.error_rate:
                phase_angle[i] = np.nan

        return timestamps, frequency, voltage, current, phase_angle, anomaly_log

    def create_dataframe(self, duration=1):

        timestamps, frequency, voltage, current, phase_angle, anomaly_log = (
            self.generate_complex_pmu_data(duration=duration)
        )
        data_frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "frequency": frequency,
                "voltage": voltage,
                "current": current,
                "phase_angle": phase_angle,
            }
        )

        # Create a column for anomaly type, initially all set to None
        data_frame["label_name"] = "Normal"
        data_frame["label"] = 0

        # Add anomaly type to the DataFrame based on the anomaly log
        for anomaly in anomaly_log:
            anomaly_time, measurement, anomaly_type = anomaly
            mask = data_frame["timestamp"] == anomaly_time
            data_frame.loc[mask, "label_name"] = f"{measurement}_{anomaly_type}"
            data_frame.loc[mask, "label"] = 1

        # Display the DataFrame with anomalies logged
        data_frame_with_anomalies = data_frame[
            [
                "timestamp",
                "frequency",
                "voltage",
                "current",
                "phase_angle",
                "label_name",
                "label",
            ]
        ]

        return data_frame_with_anomalies


# if __name__ == "__main__":
#     # Generate PMU Data and log anomalies
#     pmu_gen = PMUDataGenerator(sample_rate=60, event_rate=0.05, error_rate=0.03)
#     print(pmu_gen.create_dataframe(duration=1))
