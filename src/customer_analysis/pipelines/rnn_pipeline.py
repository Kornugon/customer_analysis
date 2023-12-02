import os
import json
import shutil
import signal
import inspect
from typing import Any, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer, AdamW
from torch.utils.data import Dataset, DataLoader

from customer_analysis.utils.mlflow_functions import MLFlowManager
from customer_analysis.utils.functional import (
    parameter_search, pytorch_classification_metrics)
from customer_analysis.models.nn import RNNModel
from customer_analysis.pipelines.nn_pipeline import (
    NNPipeline, IncorrectConfig, ModelNotFitted)


class RNNPipeline(NNPipeline):
    """
    Class implementing instance of RNNPipeline.
    """

    def __init__(
            self,
            config_path: str,
            num_classes: int,
            padding_value: int,
            model_name: str = 'RNNModel') -> None:
        """
        :param str config_path: Training params path.
        :param int num_classes: Number of classes for classification.
        :param int padding_value: The value used to pad the input sequences\
            to the same length.
        :param str model_name: Model name. Default: 'RNNModel'.
        """
        self.num_classes = num_classes
        self.padding_value = padding_value
        self.model_name = model_name

        self.model_params, self.train_params, \
            self.pipeline_params, self.grid_search_params, \
            self.mlflow_config = self._load(config_path)
        self.best_params = self.model_params

        self.task = self.train_params.get('task', 'events')
        self.device = torch.device(self.train_params.get('device', 'cpu'))
        self.loss_func = nn.CrossEntropyLoss()

        self.early_stopping_patience = self.pipeline_params.get(
            'early_stopping_patience', 5)
        self.shuffle_train_data = self.pipeline_params.get(
            'shuffle_train_dataloader', True)
        self.eval_model = self.pipeline_params.get(
            'eval_model', True)
        self.prob_thresold = self.pipeline_params.get(
            'prob_thresold', 0.5)
        self.return_churn_prob = self.pipeline_params.get(
            'return_churn_prob', False)
        self.save_attention_weights = self.pipeline_params.get(
            'save_attention_weights', True)
        self.grid_metric = self.pipeline_params.get(
            'grid_search_metric', 'accuracy')
        self.model_path = self.pipeline_params.get(
            'model_artifacts_path', 'artifacts')

        # Model __init__ parameters for filtering .json config file.
        model_init_params = inspect.signature(RNNModel.__init__)
        self.init_params = [
            param.name for param in model_init_params.parameters.values()
            if param.name != 'self']

        self.best_score = -torch.inf
        self.predicted_targets = []
        self.model_fitted = False
        self.pretrained_model_on_predict = False

        self.mlflow_manager = MLFlowManager(
            self.model_name, self.mlflow_config) \
            if self.mlflow_config['enable'] else None

        self._validate_args(self.model_params)

    def _validate_args(
            self,
            params: dict[str, Any]) -> None:
        """
        Checks the 'model_init_params' from the configuration
        .json file for validity.

        :param dict[str, Any] params: Dictionary of model parameters.

        :raises IncorrectConfig: Raises if the model cannot be initialized\
            with the provided 'model_init_params'.
        """
        params["device"] = self.device
        params["num_classes"] = self.num_classes
        params["padding_value"] = self.padding_value

        filtered_params = {
            key: params[key] for key in self.init_params
            if key in params}

        try:
            _ = RNNModel(**filtered_params)
        except Exception as exc:
            raise IncorrectConfig(
                "Cannot initialize model from 'model_init_params'\
                    see Traceback for details") from exc

    def fit(
            self,
            data: Dataset,
            validation_data: Optional[Dataset] = None) -> None:
        """
        Finds the best combination of parameters for the model.
        This method takes in a list of tensor data and performs a grid-search
        over the specified parameter grid to find the best combination
        of parameters for the model.

        :param Dataset data: Input sequences Dataset for training purpose.
        :param Optional[Dataset] validation_data: Input sequences Dataset\
            for validation purpose. Default: None.

        :raises ValueError: Raises error if training parameters\
            has not been provided.
        """
        dataloaders = {
            'train': DataLoader(
                data,
                batch_size=self.train_params['batch_size'],
                num_workers=self.train_params['num_workers'],
                shuffle=self.shuffle_train_data)
        }
        if validation_data:
            dataloaders['val'] = DataLoader(
                validation_data,
                batch_size=self.train_params['batch_size'],
                num_workers=self.train_params['num_workers'],
                shuffle=False)

        mlflow_server = self.mlflow_manager.start_mlflow_server() \
            if self.mlflow_manager else None

        self.best_model_path = \
            f'{self.model_path}/grid_model/{self.model_name}_{self.task}_task'

        phase = 'grid_search'
        if self.grid_search_params:
            grid_params = list(parameter_search(**self.grid_search_params))
            for i, params in enumerate(grid_params, start=1):
                parameters = {
                    **self.model_params, **params,
                    "num_classes": self.num_classes,
                    "padding_value": self.padding_value}
                filtered_params = {
                    key: value for key, value in parameters.items()
                    if key in self.init_params}
                model = RNNModel(**filtered_params)

                # Print grid info:
                param_grid_txt = f'\nParam grid [{i}/{len(grid_params)}]: '
                param_grid_txt += \
                    f'Attention mechanism: "{parameters["attention_type"]}".'
                param_grid_txt += \
                    f' Regularization type: "{parameters["reg_type"]}"'
                print(param_grid_txt)

                # train/val
                optimizer = AdamW(model.parameters(),
                                  parameters['learning_rate'])
                if self.mlflow_manager:
                    descr = f"{self.model_name} {self.task} grid search"
                    with self.mlflow_manager.start_run(
                            phase, self.model_name, descr):
                        scores = self._rnn_training(
                            model, dataloaders, optimizer,
                            self.train_params['num_epochs'],
                            parameters['reg_lambda'], parameters['reg_type'])
                        log_params = params | self.model_params | (
                            self.train_params if self.early_stop_epoch == -1
                            else self.train_params |
                            {"logged_early_stop_epoch": self.early_stop_epoch})
                        self.mlflow_manager.log_run(log_params, scores, model)
                else:
                    scores = self._rnn_training(
                        model, dataloaders, optimizer,
                        self.train_params['num_epochs'],
                        parameters['reg_lambda'], parameters['reg_type'])

                # Record best model, model scores & params +
                # (optionally) attention weights
                if scores[self.grid_metric] > self.best_score:
                    self.best_score = scores[self.grid_metric]
                    self.best_params = parameters | self.train_params
                    self._save_model(model, self.best_model_path + '.pth')
                    if self.save_attention_weights:
                        shutil.copyfile(
                            f'{self.best_model_path}_temporary.json',
                            f'{self.best_model_path}.json')
                if os.path.exists(f'{self.best_model_path}_temporary.json'):
                    os.remove(f'{self.best_model_path}_temporary.json')

            self.model_fitted = True
            if mlflow_server:
                os.killpg(os.getpgid(mlflow_server.pid), signal.SIGTERM)
                print("\nTerminate local MLFlow server.")
        else:
            raise ValueError("There are no training parameters provided")

    def predict(
            self,
            predict_data: Dataset,
            model_path: Optional[str] = None) -> list[int]:
        """
        Predict function.

        :param Dataset predict_data: Input sequences Dataset for prediction.
        :param Optional[str] model_path: The path to a pre-trained model\
            to be loaded. If not provided, the best model found during\
            training will be used. Default: None.

        :raises ModelNotFitted: Raises when model was not fitted\
            and path to already trained model was not provided.

        :return list[int]: The list of predicted events or churn.
        """
        phase = 'predict'
        mlflow_server = self.mlflow_manager.start_mlflow_server() \
            if self.mlflow_manager else None

        pred_dataloader = DataLoader(
            predict_data,
            batch_size=self.train_params['batch_size'],
            num_workers=self.train_params['num_workers'],
            shuffle=False)

        # Create a new instance of the model and load trained best model.
        filtered_params = {
            key: value for key, value in self.best_params.items()
            if key in self.init_params}
        model = RNNModel(**filtered_params)

        if model_path:
            model.load_state_dict(torch.load(model_path))
            self.best_model_path = f'{self.model_path}/pre_trained/'
            self.best_model_path += f'{self.model_name}_{self.task}_task'
            self.mlflow_config['tags']['used_pre_trained_model'] = 'True'
            self.pretrained_model_on_predict = True
        else:
            try:
                if self.model_fitted:
                    model.load_state_dict(
                        torch.load(self.best_model_path + '.pth'))
            except Exception as exc:
                raise ModelNotFitted(
                    "Fit model to the data first or provide 'model_path' to\
                        already trained model.") from exc

        lr = 0.001 if self.pretrained_model_on_predict \
            else self.best_params['learning_rate']
        optimizer = AdamW(model.parameters(), lr)

        if self.mlflow_manager:
            descr = f"{self.model_name} {self.task} {phase}"
            with self.mlflow_manager.start_run(
                    phase, self.model_name, descr):
                scores = self._rnn_loop(
                    'test', model, pred_dataloader, optimizer)
                progress = f"test_loss={scores['test_loss']:.3f}"
                progress += f", {self.grid_metric}="
                progress += f"{scores.get(self.grid_metric, float('nan')):.3f}"
                progress += ".\n"
                print(progress)
                self.mlflow_manager.log_run(self.best_params, scores, model)
        else:
            scores = self._rnn_loop(
                'test', model, pred_dataloader, optimizer)
            progress = f"test_loss={scores['test_loss']:.3f}"
            progress += f", {self.grid_metric}="
            progress += f"{scores.get(self.grid_metric, float('nan')):.3f}"
            progress += ".\n"
            print(progress)

        if mlflow_server:
            os.killpg(os.getpgid(mlflow_server.pid), signal.SIGTERM)
            print("\nTerminate local MLFlow server.")

        return self.predicted_targets

    def _rnn_training(
            self,
            rnn: RNNModel,
            dataloaders: dict[str, DataLoader],
            optimizer: Optimizer,
            num_epochs: int,
            reg_lambda: float = 0.0,
            reg_type: Optional[str] = None) -> dict[str, float]:
        """
        This function trains the provided model using the data loaders.
        The training process includes early stopping and model selection,
        based on the specified grid metric. Regularization can also be applied,
        using the specified regularization type.

        :param RNNModel rnn: The RNN model to be trained.
        :param dict[str, DataLoader] dataloaders: A dictionary containing\
            the data loaders for the training and validation data.
        :param Optimizer optimizer: An optimizer to use.
        :param int num_epochs: Number of epochs.
        :param str grid_metric: The additional metric to be printed.
        :param float reg_lambda: The regularization parameter. Default: 0.0.
        :param Optional[str] reg_type: The type of regularization to be used.\
            Can be ['L1', 'L2']. Default: None.
        """
        best_model_params = rnn.state_dict()
        best_val_loss = torch.inf
        epochs_without_improvement = 0
        self.early_stop_epoch = -1

        for epoch in range(num_epochs):
            metrics = defaultdict(float)
            epoch_progr = f"Epoch {epoch + 1}/{num_epochs}: "

            for phase in dataloaders.keys():
                scores = self._rnn_loop(
                    phase, rnn, dataloaders[phase], optimizer,
                    reg_lambda, reg_type, epoch_progr)
                metrics.update(scores)

            if 'val' in dataloaders:
                val_loss = metrics["val_loss"]
                if val_loss < best_val_loss:
                    best_model_params = rnn.state_dict()
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= \
                            self.early_stopping_patience:
                        self._train_val_info(epoch_progr, metrics)
                        print(f'Early stop at epoch {epoch + 1}!')
                        self.early_stop_epoch = epoch + 1
                        break

            if (epoch + 1) % self._SHOW_LOSS_INFO_STEP == 0 \
                    or epoch == num_epochs - 1:
                self._train_val_info(epoch_progr, metrics)

        rnn.load_state_dict(best_model_params)

        return metrics

    def _rnn_loop(
            self,
            phase: str,
            rnn: RNNModel,
            dataloader: DataLoader,
            optimizer: Optimizer,
            reg_lambda: float = 0.0,
            reg_type: Optional[str] = None,
            epoch_progr: Optional[str] = None) -> dict[str, float]:
        """
        This function with loop for the provided RNN model using the
        data loader and provided phase information.

        :param str phase: The phase, eg. ['train', 'val', 'test'].
        :param RNNModel rnn: The RNN model.
        :param DataLoader dataloader: The data loader for the data.
        :param Optimizer optimizer: An optimizer to use.
        :param float reg_lambda: The regularization parameter. Default: 0.0.
        :param Optional[str] reg_type: The type of regularization to be used.\
            Can be ['L1', 'L2']. Default: None.
        :param Optional[str] epoch_progr: Epoch phase string descriptor.\
            Default: None.

        :return dict[str, float]: Phase metrics.
        """
        rnn.train() if phase == 'train' else rnn.eval()
        scores = defaultdict(float)

        # Store the input sequences, true targets and predicted targets
        # for later access - eg. plotting
        true_targets, predicted_targets = [], []

        with torch.set_grad_enabled(phase == 'train'):
            iterator = tqdm(dataloader, desc=phase)
            for i, (inputs, targets, seq_lengths) in enumerate(iterator):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if not self.pretrained_model_on_predict:
                    loss, _ = rnn.step(
                        phase=phase,
                        inputs=inputs,
                        seq_lengths=seq_lengths,
                        targets=targets,
                        optimizer=optimizer,
                        loss_func=self.loss_func,
                        reg_lambda=reg_lambda,
                        reg_type=reg_type)

                    iterator.set_description(
                        epoch_progr + phase if epoch_progr else phase)

                    scores[f"{phase}_loss"] += loss

                true_targets.extend(targets.tolist())
                if self.task != 'events' and self.return_churn_prob:
                    predictions, weights = rnn.predict_prob(
                        inputs, seq_lengths)
                else:
                    predictions, weights = rnn.predict(inputs, seq_lengths)
                predicted_targets.extend(predictions.tolist())

                if self.save_attention_weights:
                    self.save_attention_data(
                        phase, i, weights, inputs,
                        predictions, rnn.attention_type)

            scores[f"{phase}_loss"] /= len(dataloader)
            predicted_targets_for_score = [
                [int(prob[1] > self.prob_thresold)]
                for prob in predicted_targets
            ] if self.task != 'events' and self.return_churn_prob \
                else predicted_targets
            if phase == 'train' and not self.eval_model:
                predicted_targets_tensor = torch.tensor(
                    predicted_targets_for_score)
                scores[self.grid_metric] = pytorch_classification_metrics(
                    labels=true_targets,
                    predictions=predicted_targets_tensor,
                    metric=self.grid_metric,
                    prob_thresold=self.prob_thresold)
            elif self.eval_model:
                if not all(element == 0 for element in true_targets):
                    scores = self.evaluate(
                        scores,
                        true_targets,
                        predicted_targets_for_score)

            if phase == 'test':
                if self.return_churn_prob and self.task != 'events':
                    self.predicted_targets = [
                        [round(prob[1], 4)] for prob in predicted_targets
                    ]
                else:
                    self.predicted_targets = predicted_targets

        return scores

    def save_attention_data(
            self,
            phase: str,
            batch_index: int,
            weights: torch.Tensor,
            batch_data: torch.Tensor,
            predictions: torch.Tensor,
            attention_type: str) -> None:
        """
        This function saves the attention weights, input data, and predictions
        for each batch to a JSON file. The data shema is as fallows:
        {
            "phase": str,
            "attention_type": str,
            "heads": {
                head_index: {
                    "batch_index": int,
                    "batch_inputs": list[list[int]],
                    "attention_weights": list[list[float]],
                    "predictions": list[int]
                    }
                }
            }

        :param str phase: The phase, eg. ['train', 'val', 'test'].
        :param int batch_index: The index of the current batch.
        :param torch.Tensor weights: The attention weights.
        :param torch.Tensor batch_data: The input data for the current batch.
        :param torch.Tensor predictions: The predictions made by the model for\
            the current batch.
        :param str attention_type: The type of attention mechanism used\
            by the model.
        """
        predictions = [int(prob[1] > self.prob_thresold)
                       for prob in predictions]\
            if self.task != 'events' and self.return_churn_prob\
            else [int(p[0]) for p in predictions]

        batch_inputs = [
            [int(event[0])
             for event in data if int(event[0]) != self.padding_value]
            for data in batch_data.tolist()]

        weights = weights.squeeze().cpu().detach().numpy().tolist()

        data = {'phase': phase, 'attention_type': attention_type, 'heads': {}}
        if attention_type == 'multi-head':
            for head_index, head in enumerate(weights):
                attention_weights = [
                    head[i][:len(sequence)]
                    for i, sequence in enumerate(batch_inputs)]

                data['heads'][f'{head_index}'] = {
                    'batch_index': batch_index,
                    'batch_inputs': batch_inputs,
                    'attention_weights': attention_weights,
                    'predictions': predictions
                }
        else:
            attention_weights = [
                weights[i][:len(sequence)]
                for i, sequence in enumerate(batch_inputs)]
            data['heads']['0'] = {
                'batch_index': batch_index,
                'batch_inputs': batch_inputs,
                'attention_weights': attention_weights,
                'predictions': predictions
            }

        if phase == 'test':
            file_pth = f'{self.best_model_path}.json'
        elif self.pretrained_model_on_predict:
            file_pth = f'{self.best_model_path}_pretrained.json'
        else:
            file_pth = f'{self.best_model_path}_temporary.json'

        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        mode = 'w' if phase == 'train' and batch_index == 0 else 'a'
        with open(file_pth, mode) as f:
            json.dump(data, f)
            f.write('\n')
