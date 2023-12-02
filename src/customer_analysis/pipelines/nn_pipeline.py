import os
import json
from abc import ABC
from typing import Any, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from customer_analysis.utils.functional import pytorch_classification_metrics


class NNPipeline(ABC):
    """
    Class implementing instance of NNPipeline as
    container with functions to inherit.
    """
    _VALID_METRICS = ['accuracy', 'precision', 'recall', 'f1']
    _SHOW_LOSS_INFO_STEP = 3

    def __init__(
            self,
            config_path: str,
            model_name: str) -> None:

        self.model_name = model_name

        self.model_params, self.train_params, \
            self.pipeline_params, self.grid_search_params, \
            self.mlflow_config = self._load(config_path)
        self.best_params = self.model_params

        self.device = torch.device(self.train_params.get('device', 'cpu'))

        self.prob_thresold = self.pipeline_params.get(
            'prob_thresold', 0.5)
        self.grid_metric = self.pipeline_params.get(
            'grid_search_metric', 'accuracy')

    def fit(
            self,
            data: Dataset) -> None:
        """
        Finds the best combination of parameters for the model.
        This method takes in a list of tensor data and performs a grid-search
        over the specified parameter grid to find the best combination
        of parameters for the model.

        :param Dataset data: Input sequences Dataset for training purpose.
        """
        pass

    def predict(
            self,
            predict_data: Dataset) -> list[int]:
        """
        Predict function.

        :param Dataset predict_data: Input sequences Dataset for prediction.

        :return list[int]: The list of predicted events or churn.
        """
        pass

    def evaluate(
            self,
            scores: dict[str, float],
            true_targets: list[Union[int, float]],
            predicted_targets: list[Union[int, float]]) -> dict[str, float]:
        """
        Score predictions.

        :param dict[str, float] scores: Scores to update.
        :param list[Union[int, float]] true_targets: True labels.
        :param list[Union[int, float]] predicted_targets: Predited labels.

        :return dict[str, float]: Updated scores.
        """
        predicted_targets_tensor = torch.tensor(predicted_targets)
        scores.update(
            {metric: pytorch_classification_metrics(
                labels=true_targets,
                predictions=predicted_targets_tensor,
                metric=metric,
                prob_thresold=self.prob_thresold)
                for metric in self._VALID_METRICS})

        return scores

    def _train_val_info(
            self,
            epoch_progr: str,
            metrics: dict[str, float]) -> None:
        """
        Helper function for printing train/val loss + metric score.
        In purpose of usage while early_stop occurs
        and after fraction of epochs.

        :param str epoch_progr: Epoch phase string descriptor.
        :param dict[str, float] metrics: Metrics scores.
        """
        progress = f"{epoch_progr}train_loss="
        progress += f"{metrics['train_loss']:.3f}"
        if 'val_loss' in metrics:
            progress += f", val_loss={metrics['val_loss']:.3f}"
        progress += f", {self.grid_metric}="
        grid_metr = metrics.get(self.grid_metric, float('nan'))
        progress += f"{grid_metr:.3f}."
        print(progress)

    def _save_model(
            self,
            model: nn.Module,
            model_path: str) -> None:
        """
        Save torch model under a path.

        :param nn.Module model: The trained model to save
        :param str model_path: Path to save the model in.
        """
        dir_name = os.path.dirname(model_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(model.state_dict(), model_path)

    def _load(self, config_path: SyntaxWarning
              ) -> tuple[dict[str, Any],
                         Union[dict[str, Any], None],
                         Union[dict[str, Any], None]]:
        if isinstance(config_path, str):
            with open(config_path, 'r') as f:
                config_params = json.load(f)
            config_params = config_params[self.model_name]
        else:
            raise TypeError(
                f"config_path should be str, but is {type(config_path)}")

        return config_params['model_init_params'], \
            config_params.get('model_training_params'), \
            config_params.get('pipeline_params'), \
            config_params.get('grid_search_params'), \
            config_params.get('mlflow_config')


class IncorrectConfig(Exception):
    """
    Cannot initialize model from 'model_init_params'.
    """


class ModelNotFitted(Exception):
    """
    Model has to be fitted first.
    """
