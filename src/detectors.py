from typing import List
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA


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


class PcaAnomalyDetector(object):
    def __init__(
        self,
        data: pd.DataFrame,
        n_components: int = 2,
        threshold: int = 3,
    ):
        self.k = threshold
        self.pca = PCA(n_components=n_components, svd_solver="full")
        self.train = self._initialzie(data)
        self.mean_distr = self.train.mean(axis=0)

    def _initialzie(self, data):
        X_train_PCA = self.pca.fit_transform(data)
        X_train_PCA = pd.DataFrame(X_train_PCA)
        X_train_PCA.index = data.index

        return X_train_PCA

    def _is_pos_def(self, A):
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def MD_threshold(self, dist, extreme=False, verbose=False):
        k = self.k if extreme else self.k - 1
        threshold = np.mean(dist) * k
        return threshold

    def MD_detectOutliers(self, dist, extreme=False, verbose=False):
        k = self.k if extreme else self.k - 1
        threshold = np.mean(dist) * k
        outliers = []
        for i in range(len(dist)):
            if dist[i] >= threshold:
                outliers.append(i)  # index of the outlier
        return np.array(outliers)

    def MahalanobisDist(self, inv_cov_matrix, mean_distr, data, verbose=False):
        inv_covariance_matrix = inv_cov_matrix
        vars_mean = mean_distr
        diff = data - vars_mean
        md = []
        for i in range(len(diff)):
            md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
        return md

    def cov_matrix(self, data, verbose=False):
        covariance_matrix = np.cov(data, rowvar=False)
        if self.is_pos_def(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if self.is_pos_def(inv_covariance_matrix):
                return covariance_matrix, inv_covariance_matrix
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
            print("Error: Covariance Matrix is not positive definite!")
