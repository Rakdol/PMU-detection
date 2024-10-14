import os
from typing import List, Union
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import Pipeline
from pickle import dump, load


# Frequency Detector
class FrequencyDetector:
    def detect(self, frequency: np.ndarray, threshold=0.2) -> np.ndarray:
        difference = np.abs(60 - frequency) > threshold
        return difference


# ROCOF Detector (Rate of Change of Frequency)
class ROCOFDetector:
    def detect(self, rocof: np.ndarray, threshold=0.0124) -> np.ndarray:
        # rocof = np.diff(frequency, prepend=frequency[0])  # Rate of Change of Frequency
        anomalies = np.abs(rocof) > threshold
        return anomalies


# Teager-Kaiser Energy Operator (TKEO) Detector
class TKEODetector:
    def detect(self, signal: np.ndarray, threshold=95) -> np.ndarray:
        teager = self.teager_kaiser_energy_operator(signal)
        upper_threshold_percentile = np.percentile(teager, threshold)
        lower_threshold_percentile = np.percentile(teager, 100 - threshold)

        upper_index = teager > upper_threshold_percentile
        lower_index = teager < lower_threshold_percentile

        anomalies_indices = upper_index | lower_index

        return anomalies_indices

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


class PcaDetector(object):
    def __init__(self, model_directory: str, model_file_name: str):
        """
        model structures
        pca_pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("ipca", IncrementalPCA(n_components=2))
        ])

        model_directory (string) : model directory foler name
        model_file_name (string) : model file name in model directory
        """
        self.model = self._get_pca_model(model_directory, model_file_name)
        self.labels = {0: "Normal", 1: "Abnormal"}
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.mean_distr = None

    def predict(self, dist: np.ndarray, threshold=3.0, extreme=True) -> np.ndarray:
        k = threshold if extreme else threshold
        threshold = np.mean(dist) * k
        outliers = []
        for i in range(len(dist)):
            if dist[i] >= threshold:
                outliers.append(i)  # index of the outlier

        return np.array(outliers)

    def predict_label(self, dist: np.ndarray, threshold=3.0, extreme=True):
        predictions = self.predict(dist, threshold, extreme)
        return np.array([self.labels[prediction] for prediction in predictions])

    def _get_pca_model(self, model_directory: str, model_file_name: str):
        model_file_directory = os.path.join(model_directory, model_file_name)
        try:
            with open(model_file_directory, "rb") as f:
                model = load(f)
        except FileNotFoundError as e:
            print(f"Model file cannot be found: {e}")
            print("Declare New  Model Pipeline")
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("ipca", IncrementalPCA(n_components=2)),
                ]
            )

        return model

    def _update(self, x: Union[pd.DataFrame, np.ndarray], scale: bool = False) -> None:
        if scale:
            # If the scaler is needed to update its parameters
            # We call fit function.
            self.model.named_steps["scaler"].fit(x)

        self.model.named_steps["scaler"].transform(x)
        self.model.named_steps["ipca"].partial_fit(x)

    def _is_pos_def(self, A: np.ndarray) -> bool:
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def _get_cov_matrix(self, data: Union[pd.DataFrame, np.ndarray]):
        covariance_matrix = np.cov(data, rowvar=False)
        if self.is_pos_def(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if self.is_pos_def(inv_covariance_matrix):
                return covariance_matrix, inv_covariance_matrix
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
            print("Error: Covariance Matrix is not positive definite!")

    def _calculate_MahalanobisDist(
        self, inv_cov_matrix: np.ndarray, mean_distr: np.ndarray, data: np.ndarray
    ):
        inv_covariance_matrix = inv_cov_matrix
        vars_mean = mean_distr
        diff = data - vars_mean
        md = []
        for i in range(len(diff)):
            md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
        return md

    def _MD_threshold(self, dist: np.ndarray, threshold=3.0, extreme=False):
        k = threshold if extreme else threshold - 1
        threshold = np.mean(dist) * k
        return threshold
