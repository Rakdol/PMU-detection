from typing import List
import pandas as pd
import numpy as np
from scipy.stats import zscore


# Frequency Detector
class FrequencyDetector:
    def detect(self, frequency: np.ndarray, threshold=0.2) -> np.ndarray:
        difference = np.abs(60 - frequency) > threshold
        return difference


# ROCOF Detector (Rate of Change of Frequency)
class ROCOFDetector:
    def detect(self, frequency: np.ndarray, threshold=0.0124) -> np.ndarray:
        rocof = np.diff(frequency, prepend=frequency[0])  # Rate of Change of Frequency
        anomalies = np.abs(rocof) > threshold
        return anomalies


# Teager-Kaiser Energy Operator (TKEO) Detector
class TKEODetector:
    def detect(self, signal: np.ndarray, threshold=0.05) -> np.ndarray:
        teager = self.teager_kaiser_energy_operator(signal)
        anomalies = teager > threshold
        return anomalies

    @staticmethod
    def teager_kaiser_energy_operator(signal: np.ndarray) -> np.ndarray:
        teager = np.zeros(len(signal))
        teager[1:-1] = signal[1:-1] ** 2 - signal[0:-2] * signal[2:]
        return teager


# Z-Score Detector
class ZscoreDetector:
    def detect(
        self, data_frame: pd.DataFrame, feature_list: List[str], threshold=3
    ) -> np.ndarray:
        z_scores = np.abs(zscore(data_frame[feature_list], nan_policy="omit"))
        anomalies = (z_scores > threshold).any(axis=1)
        return anomalies
