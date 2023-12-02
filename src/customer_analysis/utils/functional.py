from itertools import product
from typing import Any, Iterator

import torch
from torchmetrics.functional import (
    accuracy, precision, recall, fbeta_score)

_DEC_POINTS = 3


def parameter_search(
        **kwargs: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """
    The function takes keyword arguments, representing
    parameters and their possible values. Then, it generates
    all possible combinations of these parameters, using
    the Cartesian product.

    :param dict[str, list[Any]] kwargs: A dictionary of parameter names and\
        list of paremeter values.

    :return Iterator[dict[str, Any]]: An iterator that yields dictionaries\
        with parameter names and values as one possible\
        combination of parameters.
    """
    for vals in product(*kwargs.values()):
        yield dict(zip(kwargs.keys(), vals))


def pytorch_classification_metrics(
        labels: list[int],
        predictions: torch.Tensor,
        metric: str,
        beta: float = 1.0,
        proba_thresold: float = 0.5) -> float:
    """
    Calculate the specified classification metric
    for given label and prediction lists using PyTorch.

    :param list[int] labels: A list of ground truth labels.
    :param torch.Tensor predictions: A tensor of predicted probabilities\
        or a list of 'top_k' predicted labels.
    :param str metric: A string specifying the metric to calculate.
    :param float beta: The beta parameter for the F-beta score calculation.\
        Default: 1.0.
    :param float proba_thresold: The probability threshold\
        for classifying an instance as positive. Default: 0.5.

    :return float: The calculated metric score.
    """
    true_labels = torch.tensor(labels)

    # Convert predicted probabilities (positive class) to binary predictions
    if predictions[0].ndim == 2:
        predictions = (predictions[:, 1] > proba_thresold).long().unsqueeze(1)

    # Map metric names to calculation functions
    metric_functions = {
        f"accuracy": accuracy,
        f"precision": precision,
        f"recall": recall,
        f"f1": fbeta_score,
        f"f_beta": fbeta_score,
    }
    if metric not in metric_functions:
        raise ValueError(f"Invalid metric: {metric}")

    if isinstance(predictions, torch.Tensor) and predictions.ndim > 1:
        extracted_predictions = []
        for i, pred in enumerate(predictions):
            if true_labels[i] in pred:
                extracted_predictions.append(true_labels[i])
            else:
                extracted_predictions.append(pred[0])
        predictions = torch.tensor(extracted_predictions)

    # Determine number of classes and task
    num_classes = (max(
        max(true_labels.int()), max(predictions.int()))).item() + 1
    task = 'binary' if num_classes == 2 else 'multiclass'
    average = 'micro'

    # Calculate specified metric for input lists
    if metric == "f_beta":
        result = round(metric_functions[metric](
            predictions, true_labels,
            task=task, average=average, num_classes=num_classes,
            beta=beta).item(), _DEC_POINTS)
    else:
        result = round(metric_functions[metric](
            predictions, true_labels,
            task=task, average=average, num_classes=num_classes)
            .item(), _DEC_POINTS)

    return result
