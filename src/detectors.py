import os
from typing import List, Union
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import Pipeline
from pickle import dump, load
from src.utils import logger


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
        PCA 모델을 초기화하거나 새롭게 생성하는 클래스
        model_directory (string): 모델이 저장된 디렉터리 경로
        model_file_name (string): 모델 파일 이름
        """
        self.model = self._get_pca_model(model_directory, model_file_name)
        self.labels = {0: "Normal", 1: "Abnormal"}
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.mean_distr = None
        self.prior_mean = None
        self.prior_var = None
        self.threshold = 3.0
        self.update_threshold = 5.0  # 업데이트 여부를 판단할 임계값 설정

    def _get_pca_model(self, model_directory: str, model_file_name: str):
        model_file_directory = os.path.join(model_directory, model_file_name)
        try:
            with open(model_file_directory, "rb") as f:
                model = load(f)
        except FileNotFoundError as e:
            logger.info(f"Model file cannot be found: {e}")
            logger.info("Declare New Model Pipeline with Incremental PCA")
            # Incremental PCA로 새로운 모델을 생성
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),  # 스케일러로 입력 데이터 표준화
                    ("ipca", IncrementalPCA(n_components=2)),  # Incremental PCA 사용
                ]
            )
        return model

    def save_model(self, model_directory: str, model_file_name: str) -> None:
        model_file_directory = os.path.join(model_directory, model_file_name)
        with open(model_file_directory, "wb") as f:
            dump(self.model, f)

    def load_model(self, model_directory: str, model_file_name: str) -> None:
        model_file_directory = os.path.join(model_directory, model_file_name)
        try:

            with open(model_file_directory, "rb") as f:
                model = load(f)
        except FileNotFoundError as e:
            print(f"Model file cannot be found: {e}")

        self.model = model

    def _should_update(self, data: np.ndarray) -> bool:
        """
        새로운 데이터가 기존 데이터와 너무 다르면 업데이트를 하지 않음
        """

        if self.inv_cov_matrix is None or self.mean_distr is None:
            return True  # 모델이 초기화되지 않았으면 업데이트 허용

        # Step 2: 데이터를 PCA로 변환
        x_scaled = self.model.named_steps["scaler"].transform(data)
        x_pca = self.model.named_steps["ipca"].transform(x_scaled)

        mahalanobis_distances = self._calculate_MahalanobisDist(
            self.inv_cov_matrix, self.mean_distr, x_pca
        )
        # 마할라노비스 거리가 임계값을 넘으면 업데이트하지 않음
        if np.mean(mahalanobis_distances) > self.update_threshold:
            print(
                "New data significantly deviates from the existing data. Skipping update."
            )
            return False
        return True

    def update(self, x: Union[pd.DataFrame, np.ndarray], scale: bool = False) -> None:
        if not self._should_update(x):
            return  # 업데이트하지 않음

        if scale:
            # 매번 fit 대신 partial_fit으로 이전 데이터를 유지한 채 업데이트
            self.model.named_steps["scaler"].partial_fit(x)

        x_scaled = self.model.named_steps["scaler"].transform(x)
        # Incremental PCA를 통해 점진적으로 학습
        self.model.named_steps["ipca"].partial_fit(x_scaled)

        # PCA 변환 결과를 얻음
        x_pca = self.model.named_steps["ipca"].transform(x_scaled)

        # 공분산 행렬 및 평균 벡터 업데이트
        self.cov_matrix, self.inv_cov_matrix = self._get_cov_matrix(x_pca)
        self.mean_distr = np.mean(x_pca, axis=0)

    def _get_cov_matrix(self, data: Union[pd.DataFrame, np.ndarray]):
        covariance_matrix = np.cov(data, rowvar=False)
        inv_cov_matrix = np.linalg.inv(covariance_matrix)
        return covariance_matrix, inv_cov_matrix

    def _calculate_MahalanobisDist(
        self, inv_cov_matrix: np.ndarray, mean_distr: np.ndarray, data: np.ndarray
    ):
        """
        PCA로 변환된 데이터와 평균 벡터를 사용하여 마할라노비스 거리 계산
        """
        diff = data - mean_distr
        md = []
        for i in range(len(diff)):
            md.append(np.sqrt(diff[i].dot(inv_cov_matrix).dot(diff[i])))
        return np.array(md)

    def predict(self, dist: np.ndarray, threshold) -> np.ndarray:

        outliers = np.where(
            dist >= threshold, True, False
        )  # 조건을 만족하는 인덱스 반환

        return outliers

    def _MD_threshold(self, dist: np.ndarray, threshold=3.0, extreme=False):
        """
        베이지안 방식으로 임계값을 업데이트하는 함수
        """
        # 새로운 데이터의 평균과 분산
        new_mean = np.mean(dist)
        new_var = np.var(dist)

        # 이전 평균과 분산이 없는 경우 초기화
        if self.prior_mean is None or self.prior_var is None:
            updated_mean = new_mean  # 새로운 데이터의 평균을 업데이트된 값으로 설정
            updated_var = new_var  # 새로운 데이터의 분산을 업데이트된 값으로 설정
            self.prior_mean = updated_mean  # 이후 참조를 위해 저장
            self.prior_var = updated_var  # 이후 참조를 위해 저장
        else:
            # 가중치를 적용하여 업데이트 (이전 데이터의 크기를 고려)
            weight_prior = 0.5  # 이전 평균 가중치
            weight_new = 0.5  # 새로운 평균 가중치
            updated_mean = (weight_prior * self.prior_mean) + (weight_new * new_mean)
            updated_var = (weight_prior * self.prior_var) + (weight_new * new_var)

            # 이전 상태 업데이트
            self.prior_mean = updated_mean
            self.prior_var = updated_var

        # 업데이트된 임계값 계산
        k = threshold if extreme else threshold - 1
        dynamic_threshold = updated_mean + k * np.sqrt(updated_var)

        return dynamic_threshold

    def process_and_detect(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        extreme=False,
    ) -> np.ndarray:
        """
        새로운 데이터를 처리하고 PCA 모델을 업데이트한 뒤 이상 탐지 수행
        """
        # Step 1: PCA 모델 업데이트
        if self.prior_mean is None or self.prior_var is None:
            self.update(data, scale=True)

        # Step 2: 데이터를 PCA로 변환
        x_scaled = self.model.named_steps["scaler"].transform(data)
        x_pca = self.model.named_steps["ipca"].transform(x_scaled)

        # Step 3: 마할라노비스 거리 계산
        mahalanobis_distances = self._calculate_MahalanobisDist(
            self.inv_cov_matrix, self.mean_distr, x_pca  # 변환된 PCA 데이터 사용
        )
        self.theshold = self._MD_threshold(
            mahalanobis_distances, threshold=self.threshold, extreme=extreme
        )

        # Step 4: 이상치 탐지
        outliers = self.predict(mahalanobis_distances, self.threshold)
        self.update(data)
        return outliers
