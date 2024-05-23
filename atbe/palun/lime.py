from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import Lasso


class LimeExplainer(object):
    def __init__(self):
        pass

    def _lime_explanation(
        self,
        local_data: pd.DataFrame,
        predictions: np.ndarray,
        instance: pd.Series,
        num_features: int,
        max_iter: int,
        fit_intercept: bool,
        alpha: float,
    ) -> np.ndarray:
        # Ensure local_data and instance are numpy arrays of type float
        if not isinstance(local_data, pd.DataFrame):
            raise ValueError("local_data must be a pandas DataFrame")
        if not isinstance(instance, pd.Series):
            raise ValueError("instance must be a pandas Series")
        if num_features != local_data.shape[1]:
            raise ValueError(
                "Length of features list must match the number of columns in local_data"
            )
        if isinstance(local_data, pd.DataFrame):
            local_data = local_data.values.astype(float)
        if isinstance(instance, pd.Series):
            instance = instance.values.astype(float)

        # Step 1: Compute distances (weights) based on proximity to the instance
        distances = np.sqrt(np.sum((local_data - instance) ** 2, axis=1))
        weights = np.exp(-(distances**2) / np.std(distances))

        # Step 2: Train a weighted, interpretable model (Lasso)
        explanation_model = Lasso(
            alpha=alpha, max_iter=max_iter, fit_intercept=fit_intercept
        )
        explanation_model.fit(local_data, predictions, sample_weight=weights)

        # Step 3: Reduce the number of features to num_features using feature selection
        # Here, we use Lasso's inherent feature selection
        # Retrain if necessary to adjust the number of features
        while np.count_nonzero(explanation_model.coef_) > num_features:
            alpha += 0.01  # Increase regularization
            explanation_model = Lasso(
                alpha=alpha, max_iter=max_iter, fit_intercept=fit_intercept
            )
            explanation_model.fit(local_data, predictions, sample_weight=weights)

        return explanation_model.coef_

    def _plot_coefficients(self, features, coef, ylabel, xlabel, title, filename=None):
        plt.figure(figsize=(10, 6))
        colors = ["red" if x < 0 else "blue" for x in coef]
        bars = plt.barh(features, coef, color=colors)
        plt.axvline(0, color="gray", linewidth=0.8)

        for bar, value in zip(bars, coef):
            plt.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="right" if value < 0 else "left",
            )

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            display(plt.gcf())
            plt.close()

    def show_lime_explanation(
        self,
        local_data: pd.DataFrame,
        predictions: np.ndarray,
        instance: pd.Series,
        features: List[str],
        max_iter: int = 1000,
        fit_intercept: bool = True,
        alpha: float = 0.1,
        ylabel: str = "Features",
        xlabel: str = "Coefficient Value",
        title: str = None,
    ):

        num_features = len(features)
        coef = self._lime_explanation(
            local_data,
            predictions,
            instance,
            num_features,
            max_iter,
            fit_intercept,
            alpha,
        )

        self._plot_coefficients(features, coef, ylabel, xlabel, title)

    def save_lime_as_file(
        self,
        filename: str,
        local_data: pd.DataFrame,
        predictions: np.ndarray,
        instance: pd.Series,
        features: List[str],
        max_iter: int = 1000,
        fit_intercept: bool = True,
        alpha: float = 0.1,
        ylabel: str = "Features",
        xlabel: str = "Coefficient Value",
        title: str = None,
    ):

        num_features = len(features)
        coef = self._lime_explanation(
            local_data,
            predictions,
            instance,
            num_features,
            max_iter,
            fit_intercept,
            alpha,
        )

        self._plot_coefficients(features, coef, ylabel, xlabel, title, filename)
