from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from pandas import Series
from sklearn.metrics import confusion_matrix


class ConfusionMatrix(object):
    """
    A class for generating and visualizing confusion matrices.
    """

    def __init__(self):
        pass

    def _validate_inputs(
        self, music_class: Series, prediction: Series, rotation: int
    ) -> None:
        """
        Validates the input parameters for generating a confusion matrix.

        Args:
            music_class (Series): Series containing the true classifications.
            prediction (Series): Series containing the predicted classifications.
            rotation (int): Degree of rotation for labels on the plot.

        Raises:
            ValueError: If either music_class or prediction is not a pandas Series.
            ValueError: If either music_class or prediction Series is empty.
            ValueError: If music_class and prediction Series do not have the same length.
            ValueError: If rotation is not an integer between 0 and 360 degrees.
        """
        if not isinstance(music_class, Series) or not isinstance(prediction, Series):
            raise ValueError("music_class and prediction must be pandas Series.")
        if music_class.empty or prediction.empty:
            raise ValueError("Input series cannot be empty.")
        if len(music_class) != len(prediction):
            raise ValueError(
                "Music class and prediction series must have the same length."
            )
        if not isinstance(rotation, int) or not (0 <= rotation <= 360):
            raise ValueError("Rotation must be an integer between 0 and 360 degrees.")

    def _plot_heatmap(
        self,
        data: np.ndarray,
        labels: list,
        rotation: int,
        plt_title: str,
        plt_xlabel: str,
        plt_ylabel: str,
        fontsize: int,
        figsize: Tuple[int, int],
    ) -> None:
        """
        Plots a heatmap for the given confusion matrix data.

        Args:
            data (np.ndarray): The confusion matrix data to be visualized.
            labels (list): The labels for the confusion matrix axes.
            rotation (int): The rotation angle for the axis labels.
            plt_title (str): The title of the plot.
            plt_xlabel (str): The label for the x-axis.
            plt_ylabel (str): The label for the y-axis.
            fontsize (int): The font size for text elements in the plot.
            figsize (Tuple[int, int]): The dimensions of the figure (width, height).
        """
        plt.figure(figsize=figsize)
        sns.heatmap(
            data,
            annot=True,
            fmt="g",
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={"size": fontsize},
        )
        plt.xticks(rotation=rotation, fontsize=fontsize)
        plt.yticks(rotation=rotation, fontsize=fontsize)
        plt.title(plt_title)
        plt.xlabel(plt_xlabel, fontsize=fontsize)
        plt.ylabel(plt_ylabel, fontsize=fontsize)

    def _plot_confusion_matrix(
        self,
        music_class: Series,
        prediction: Series,
        rotation: int,
        plt_title: str,
        plt_xlabel: str,
        plt_ylabel: str,
        fontsize: int = 12,
        figsize: Tuple[int, int] = (10, 7),
    ) -> None:
        """
        Generates and plots a confusion matrix using seaborn heatmap.

        Args:
            music_class (Series): Series containing the true class labels.
            prediction (Series): Series containing the predicted class labels.
            rotation (int): Degree of rotation for the labels on the x and y axis.
            plt_title (str): Title of the plot.
            plt_xlabel (str): Label for the x-axis.
            plt_ylabel (str): Label for the y-axis.
            fontsize (int, optional): Font size for the labels and title. Defaults to 12.
            figsize (Tuple[int, int], optional): Size of the figure as a tuple (width, height). Defaults to (10, 7).

        Raises:
            ValueError: If either music_class or prediction series is empty.
            ValueError: If the lengths of music_class and prediction series do not match.
            ValueError: If rotation is not an integer or is not between 0 and 360 degrees.
        """
        self._validate_inputs(music_class, prediction, rotation)
        labels = np.unique(np.concatenate((music_class, prediction)))
        cm = confusion_matrix(music_class, prediction, labels=labels)
        self._plot_heatmap(
            cm, labels, rotation, plt_title, plt_xlabel, plt_ylabel, fontsize, figsize
        )

    def _plot_subtracted_confusion_matrix(
        self,
        target_classes: list[str],
        target_confusion_matrix: np.ndarray,
        music_class: Series,
        prediction: Series,
        rotation: int,
        plt_title: str,
        plt_xlabel: str,
        plt_ylabel: str,
        fontsize: int = 12,
        figsize: Tuple[int, int] = (10, 7),
    ) -> None:
        """
        Generates and plots a subtracted confusion matrix using seaborn heatmap.

        Args:
            target_classes (list[str]): List of target class labels.
            target_confusion_matrix (np.ndarray): Target confusion matrix to subtract from the actual confusion matrix.
            music_class (Series): Series containing the true class labels.
            prediction (Series): Series containing the predicted class labels.
            rotation (int): Degree of rotation for the labels on the x and y axis.
            plt_title (str): Title of the plot.
            plt_xlabel (str): Label for the x-axis.
            plt_ylabel (str): Label for the y-axis.
            fontsize (int, optional): Font size for the labels and title. Defaults to 12.
            figsize (Tuple[int, int], optional): Size of the figure as a tuple (width, height). Defaults to (10, 7).

        Raises:
            ValueError: If either music_class or prediction series is empty.
            ValueError: If the lengths of music_class and prediction series do not match.
            ValueError: If rotation is not an integer or is not between 0 and 360 degrees.
            ValueError: If target_confusion_matrix is not a numpy array or does not have the correct shape.
        """
        self._validate_inputs(music_class, prediction, rotation)
        if not isinstance(
            target_confusion_matrix, np.ndarray
        ) or target_confusion_matrix.shape != (
            len(target_classes),
            len(target_classes),
        ):
            raise ValueError(
                "target_confusion_matrix must be a numpy array with shape (len(target_classes), len(target_classes))."
            )
        cm = confusion_matrix(music_class, prediction, labels=target_classes)
        subtracted_cm = np.subtract(cm, target_confusion_matrix)
        self._plot_heatmap(
            subtracted_cm,
            target_classes,
            rotation,
            plt_title,
            plt_xlabel,
            plt_ylabel,
            fontsize,
            figsize,
        )

    def visualize_in_notebook(
        self,
        music_class: Series,
        prediction: Series,
        rotation: int = 45,
        plt_title: str = None,
        plt_xlabel: str = "Prediction",
        plt_ylabel: str = "Class",
        fontsize: int = 12,
        figsize: Tuple[int, int] = (10, 7),
    ) -> None:
        """
        Displays the confusion matrix in the current Jupyter notebook.

        Args:
            music_class (Series): Series containing the true class labels.
            prediction (Series): Series containing the predicted class labels.
            rotation (int, optional): Degree of rotation for the labels on the x and y axis. Defaults to 45.
            plt_title (str, optional): Title of the plot. Defaults to None.
            plt_xlabel (str, optional): Label for the x-axis. Defaults to "Prediction".
            plt_ylabel (str, optional): Label for the y-axis. Defaults to "Class".
            fontsize (int, optional): Font size for the labels and title. Defaults to 12.
            figsize (Tuple[int, int], optional): Size of the figure as a tuple (width, height). Defaults to (10, 7).
        """
        self._plot_confusion_matrix(
            music_class,
            prediction,
            rotation,
            plt_title,
            plt_xlabel,
            plt_ylabel,
            fontsize,
            figsize,
        )
        display(plt.gcf())
        plt.close()

    def visualize_subtracted_in_notebook(
        self,
        target_classes: list[str],
        target_confusion_matrix: np.ndarray,
        music_class: Series,
        prediction: Series,
        rotation: int = 45,
        plt_title: str = None,
        plt_xlabel: str = "Prediction",
        plt_ylabel: str = "Class",
        fontsize: int = 12,
        figsize: Tuple[int, int] = (10, 7),
    ) -> None:
        self._plot_subtracted_confusion_matrix(
            target_classes,
            target_confusion_matrix,
            music_class,
            prediction,
            rotation,
            plt_title,
            plt_xlabel,
            plt_ylabel,
            fontsize,
            figsize,
        )
        display(plt.gcf())
        plt.close()

    def save_as_file(
        self,
        filename: str,
        music_class: Series,
        prediction: Series,
        rotation: int = 45,
        plt_title: str = None,
        plt_xlabel: str = "Prediction",
        plt_ylabel: str = "Class",
        fontsize: int = 12,
        figsize: Tuple[int, int] = (10, 7),
    ) -> None:
        """
        Saves the confusion matrix as a file.

        Args:
            filename (str): Name of the file to save the plot as.
            music_class (Series): Series containing the true class labels.
            prediction (Series): Series containing the predicted class labels.
            rotation (int, optional): Degree of rotation for the labels on the x and y axis. Defaults to 45.
            plt_title (str, optional): Title of the plot. Defaults to None.
            plt_xlabel (str, optional): Label for the x-axis. Defaults to "Prediction".
            plt_ylabel (str, optional): Label for the y-axis. Defaults to "Class".
            fontsize (int, optional): Font size for the labels and title. Defaults to 12.
            figsize (Tuple[int, int], optional): Size of the figure as a tuple (width, height). Defaults to (10, 7).
        """
        self._plot_confusion_matrix(
            music_class,
            prediction,
            rotation,
            plt_title,
            plt_xlabel,
            plt_ylabel,
            fontsize,
            figsize,
        )
        plt.savefig(filename)
        plt.close()

    def save_subtracted_as_file(
        self,
        filename: str,
        target_classes: list[str],
        target_confusion_matrix: np.ndarray,
        music_class: Series,
        prediction: Series,
        rotation: int = 45,
        plt_title: str = None,
        plt_xlabel: str = "Prediction",
        plt_ylabel: str = "Class",
        fontsize: int = 12,
        figsize: Tuple[int, int] = (10, 7),
    ) -> None:
        """
        Saves the subtracted confusion matrix as a file.

        Args:
            filename (str): Name of the file to save the plot as.
            target_classes (list[str]): List of target class labels.
            target_confusion_matrix (np.ndarray): Target confusion matrix to subtract from the actual confusion matrix.
            music_class (Series): Series containing the true class labels.
            prediction (Series): Series containing the predicted class labels.
            rotation (int, optional): Degree of rotation for the labels on the x and y axis. Defaults to 45.
            plt_title (str, optional): Title of the plot. Defaults to None.
            plt_xlabel (str, optional): Label for the x-axis. Defaults to "Prediction".
            plt_ylabel (str, optional): Label for the y-axis. Defaults to "Class".
            fontsize (int, optional): Font size for the labels and title. Defaults to 12.
            figsize (Tuple[int, int], optional): Size of the figure as a tuple (width, height). Defaults to (10, 7).
        """
        self._plot_subtracted_confusion_matrix(
            target_classes,
            target_confusion_matrix,
            music_class,
            prediction,
            rotation,
            plt_title,
            plt_xlabel,
            plt_ylabel,
            fontsize,
            figsize,
        )
        plt.savefig(filename)
        plt.close()
