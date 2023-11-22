import os
import json
import inspect
from datetime import datetime
from collections import defaultdict
from typing import Any, Optional
import signal

import mlflow
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import Dataset, DataLoader

from customer_analysis.pipelines.nn_pipeline import NNPipeline, \
    IncorrectConfig, ModelNotFitted
from customer_analysis.models.nn import RNNModel
from customer_analysis.utils.functional import parameter_search, \
    pytorch_classification_metrics


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
        :param int padding_value: The value used to pad the input sequences \
            to the same length.
        :param str model_name: Model name. Default: 'RNNModel'.
        """
        self.num_classes = num_classes
        self.padding_value = padding_value
        self.model_name = model_name

        self.model_params, self.train_params, self.grid_search_params, \
            self.mlflow_config = self.load(config_path)
        self.best_params = self.model_params

        self.device = torch.device(self.train_params.get('device', 'cpu'))
        self.loss_func = nn.CrossEntropyLoss()
        self.eval_model = self.train_params['eval_model']
        self.task = self.train_params.get('task', 'events')

        self.grid_metric = self.train_params.get(
            'grid_search_metric', 'accuracy')
        self.proba_thresold = self.train_params.get(
            'proba_thresold', 0.5)
        self.return_churn_prob = self.train_params.get(
            'return_churn_prob', False)

        # Model __init__ parameters for filtering .json config file.
        model_init_params = inspect.signature(RNNModel.__init__)
        self.init_params = [
            param.name for param in model_init_params.parameters.values()
            if param.name != 'self']

        self.save_attention_weights = self.train_params.get(
            'save_attention_weights', True)
        self.best_model_path = self.train_params.get(
            'model_artifacts_path', 'artifacts')
        self.complete_model_path = ''
        self.predicted_targets = []
        self.model_fitted = False
        self.mlflow_server = None

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
                "Cannot initialize model from 'model_init_params' \
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

        :param list[torch.Tensor] sequences: Input sequences list of tensors.
        :param Optional[list[torch.Tensor]] targets: List of target tensors. \
            Default: None.

        :raises ValueError: Raises error mlflow_config is not \
            present or lacks of "enable".
        :raises ValueError: Raises error if training parameters \
            has not been provided.
        """
        dataloaders = {
            'train': DataLoader(
                data,
                batch_size=self.train_params['batch_size'],
                num_workers=self.train_params['num_workers'],
                shuffle=self.train_params['shuffle_train_dataloader'])
        }
        if validation_data:
            dataloaders['val'] = DataLoader(
                validation_data,
                batch_size=self.train_params['batch_size'],
                num_workers=self.train_params['num_workers'],
                shuffle=False)

        models_results, grid_scores = [], []
        if self.grid_search_params:
            if self.mlflow_config['enable'] and \
                    self.mlflow_config['use_local_server']:
                local_port = self.mlflow_config['local_port']
                self.mlflow_server = self.start_mlflow_server(port=local_port)
                mlflow.set_tracking_uri(f"http://localhost:{local_port}")

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

                # model artifacts path
                self.complete_model_path = \
                    f'{self.best_model_path}/{self.model_name}'
                self.complete_model_path += f'/grid_model_{i}/model_{i}'

                # train/val
                optimizer = AdamW(model.parameters(),
                                  parameters['learning_rate'])
                scores = self._rnn_training(
                    model, dataloaders, optimizer,
                    self.train_params['num_epochs'],
                    parameters['reg_lambda'], parameters['reg_type'])

                self.save_model(model, f'{self.complete_model_path}.pth')
                grid_scores.append(
                    (params | self.model_params | self.train_params | scores))
                models_results.append({
                    'model_id': i,
                    'model_path': f'{self.complete_model_path}.pth',
                    'params': parameters | self.train_params,
                    **scores})

                # if the mlfow_config key is provided, start logging
                try:
                    if self.mlflow_config['enable']:
                        run_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
                        with mlflow.start_run(
                                run_name=run_time,
                                tags={'phase': 'grid_search'},
                                description=f"""
                                    {self.model_name} {self.task} grid search
                                """):
                            mlflow.log_params(
                                params | self.model_params | self.train_params)
                            mlflow.log_metrics(scores)
                except KeyError:
                    raise ValueError("No mlflow_config provided or \
                                     enable value in config file")

            grid_logs = pd.DataFrame(models_results) \
                .sort_values(self.grid_metric, ascending=False)
            self.best_params = grid_logs.iloc[0]['params']
            self.best_model_path = grid_logs.iloc[0]['model_path']
            self.model_fitted = True
            if self.mlflow_server:
                os.killpg(os.getpgid(self.mlflow_server.pid), signal.SIGTERM)
                print("\nTerminate local MLFlow server.")
        else:
            raise ValueError("There are no training parameters provided")

    def predict(
            self,
            predict_data: Dataset,
            model_path: Optional[str] = None) -> list[int]:
        """
        Predict function.

        :param list[torch.Tensor] predict_data: Input sequences.
        :param Optional[str] model_path: The path to a pre-trained model \
            to be loaded. If not provided, the best model found during \
            training will be used. Default: None.

        :raises ModelNotFitted: Raises when model was not fitted \
            and path to already trained model was not provided.

        :return list[int]: The list of predicted events or churn.
        """
        if self.mlflow_config['enable'] and \
                self.mlflow_config['use_local_server']:
            local_port = self.mlflow_config['local_port']
            self.mlflow_server = self.start_mlflow_server(port=local_port)
            mlflow.set_tracking_uri(f"http://localhost:{local_port}")

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
        else:
            try:
                if self.model_fitted:
                    model.load_state_dict(torch.load(self.best_model_path))
                    # exclude :-4 as for .pth
                    self.complete_model_path = self.best_model_path[:-4]
            except Exception as exc:
                raise ModelNotFitted(
                    "Fit model to the data first or provide 'model_path' to \
                        already trained model.") from exc

        optimizer = AdamW(model.parameters(),
                          self.best_params['learning_rate'])
        scores = self._rnn_loop(
            'test', model, pred_dataloader, optimizer)

        progress = f"test_loss={scores['test_loss']:.3f}"
        progress += f", {self.grid_metric}="
        progress += f"{scores.get(self.grid_metric, float('nan')):.3f}."
        print(progress)

        # if the mlfow_config key is provided, start logging
        try:
            if self.mlflow_config['enable']:
                run_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
                with mlflow.start_run(
                        run_name=run_time,
                        tags={'phase': 'predicting'},
                        description=f"""
                            {self.model_name} {self.task} prediction
                        """):
                    mlflow.log_params(self.best_params)
                    mlflow.log_metrics(scores)
        except KeyError:
            raise ValueError(
                "No mlflow_config provided or enable value in config file")

        if self.mlflow_server:
            os.killpg(os.getpgid(self.mlflow_server.pid), signal.SIGTERM)
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
                            self.train_params['early_stopping_patience']:
                        self.train_val_info(epoch_progr, metrics)
                        print(f'Early stop at epoch {epoch + 1}!')
                        break

            if (epoch + 1) % self._SHOW_LOSS_INFO_STEP == 0 \
                    or epoch == num_epochs - 1:
                self.train_val_info(epoch_progr, metrics)

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
        :param Optional[str] epoch_progr: Epoch phase string descriptor. \
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

                loss, weights = rnn.step(
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
                    predictions = rnn.predict_proba(
                        inputs, seq_lengths)
                else:
                    predictions = rnn.predict(inputs, seq_lengths)
                predicted_targets.extend(predictions.tolist())

                if self.save_attention_weights:
                    self.save_attention_data(
                        phase, i, weights, inputs,
                        predictions, rnn.attention_type)

            scores[f"{phase}_loss"] /= len(dataloader)
            predicted_targets_for_score = [
                [int(proba[1] > self.proba_thresold)]
                for proba in predicted_targets
            ] if self.task != 'events' and self.return_churn_prob \
                else predicted_targets
            if phase == 'train' and not self.eval_model:
                predicted_targets_tensor = torch.tensor(
                    predicted_targets_for_score)
                scores[self.grid_metric] = pytorch_classification_metrics(
                    labels=true_targets,
                    predictions=predicted_targets_tensor,
                    metric=self.grid_metric,
                    proba_thresold=self.proba_thresold)
            elif self.eval_model:
                if not all(element == 0 for element in true_targets):
                    scores = self.evaluate(
                        scores,
                        true_targets,
                        predicted_targets_for_score)

            if phase == 'test':
                if self.return_churn_prob and self.task != 'events':
                    self.predicted_targets = [
                        [round(proba[1], 4)] for proba in predicted_targets
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
        predictions = [int(proba[1] > self.proba_thresold)
                       for proba in predictions]\
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

        os.makedirs(os.path.dirname(self.complete_model_path), exist_ok=True)
        mode = 'w' if phase == 'train' and batch_index == 0 else 'a'
        with open(f'{self.complete_model_path}.json', mode) as f:
            json.dump(data, f)
            f.write('\n')
