from abc import ABC
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from customer_analysis.models.nn._transformer import (
    TransformerEncoderLayer, TransformerEncoder, PositionalEncoder)


class TransformerModel(nn.Module, ABC):
    """
    The model is used to create a Transformer Neural Network with options
    to predict the next event or churn based on sequences of events
    as previous actions of a customer.
    """
    _REG_NORMS = {'L1': 1, 'L2': 2}
    _types_validation = {'input_size': int,
                         'model_size': int,
                         'num_layers': int,
                         'dim_feedforward': int,
                         'num_heads': int,
                         'dropout_rate': float}

    def __init__(
            self,
            input_size: int,
            model_size: int,
            num_layers: int,
            dim_feedforward: int,
            device: torch.device,
            num_heads: int = 2,
            dropout_rate: float = 0.1,
            weights_init_range: list[float] = [-0.1, 0.1],
            task: str = 'events') -> None:
        """
        :param int input_size: The number of expected features in the input.
        :param int model_size: The dimension of the model.
        :param int num_layers: The number of encoder layers.
        :param int dim_feedforward: The dimension of the feedforward network.
        :param torch.device device: Selected compute device from\
            [`cpu`, `cuda`].
        :param int num_heads: The number of attention heads. Default: 2.
        :param float dropout_rate: The dropout rate. Default: 0.1.
        :param list[float] weights_init_range: The range for uniform\
            distribution from which, weights are initialized.\
            Default: [-0.1, 0.1].
        :param str task: The task for which the model is being used.\
            Can be either 'churn' for binary classification or 'events'\
            for event prediction. Default: 'events'.

        Input shape
            - A 2D tensor with shape (batch_size, seq_len).

        Output shape
            - A 3D tensor with shape (batch_size, seq_len, input_size).
        """
        super().__init__()
        self._validate_args(input_size=input_size,
                            model_size=model_size,
                            num_layers=num_layers,
                            dim_feedforward=dim_feedforward,
                            num_heads=num_heads,
                            dropout_rate=dropout_rate)

        self.device = device
        self.task = task
        self.scaling = torch.sqrt(torch.tensor(model_size).float())

        self.embedding = nn.Embedding(input_size, model_size)
        self.positioning = PositionalEncoder(model_size, dropout_rate)

        encoder_layer = TransformerEncoderLayer(
            model_size, num_heads, dim_feedforward,
            dropout=dropout_rate, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(
            model_size, input_size) if self.task == 'events' \
            else nn.Linear(model_size, 2)

        self.init_weights(weights_init_range)
        self.to(self.device)

    def init_weights(
            self,
            init_range: list[float]) -> None:
        """
        Initialize the weights of the embedding and fully
        connected layers. The weights are initialized from a uniform
        distribution between init_range values. The bias of the
        fully connected layer is initialized to zero.

        :param list[float] init_range: The range for uniform distribution\
            from which, weights are initialized.
        """
        self.embedding.weight.data.uniform_(init_range[0], init_range[1])
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range[0], init_range[1])

    def forward(
            self,
            seq: torch.Tensor,
            inputs_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Embedding layer is scaled by the square root of the model size
        before being passed to the transformer layer.

        :param torch.Tensor seq: The input sequence data.
        :param torch.Tensor inputs_mask: A boolean mask specifying which\
            elements in the input sequences are padded and should be ignored.

        :return tuple[torch.Tensor, list[torch.Tensor]]: Tuple of the model\
            outputs and attention weights.
        """
        x = self.embedding(seq) * self.scaling
        x = self.positioning(x)
        x, w = self.transformer(
            x, src_key_padding_mask=inputs_mask)

        last_real_token_idx = ((~inputs_mask).sum(axis=1) - 1)[:, None, None]
        x = torch.take_along_dim(x, last_real_token_idx, dim=1)

        x = self.fc(x)

        return x, w

    def predict(
            self,
            sequence: torch.Tensor,
            inputs_mask: torch.Tensor,
            top_k: int = 1) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generates a prediction for the next event in a sequence of events
        or churn based on sequence of events.

        :param torch.Tensor sequence: The input data.
        :param torch.Tensor inputs_mask: A boolean mask that indicates\
            true and padded values.
        :param int top_k: The number of top results to return. Default: 1.

        :return tuple[torch.Tensor, list[torch.Tensor]]: The predicted index\
            of event or churn and attention weights.
        """
        output, attention_weights = self(sequence, inputs_mask)
        _, top_k_indices = torch.topk(output[:, -1], top_k, dim=-1)

        return top_k_indices, attention_weights

    def predict_proba(
            self,
            sequence: torch.Tensor,
            inputs_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generates the probabilities for each class for the churn
        based on sequence of events.

        :param torch.Tensor sequence: The input data.
        :param torch.Tensor inputs_mask: A boolean mask that indicates\
            true and padded values.

        :return tuple[torch.Tensor, list[torch.Tensor]]: A tensor of shape\
            (batch_size, num_classes) containing the probabilities for each\
            class for each input sequence and attention weights.
        """
        output, attention_weights = self(sequence, inputs_mask)
        probabilities = torch.softmax(output[:, -1], dim=-1)

        return probabilities, attention_weights

    def step(
            self,
            phase: str,
            inputs: torch.Tensor,
            inputs_mask: torch.Tensor,
            targets: torch.Tensor,
            loss_func: nn.Module,
            optimizer: Optimizer,
            reg_lambda: float = 0.0,
            reg_type: Optional[str] = None) -> tuple[float, torch.Tensor]:
        """
        Performs one step of training on the given inputs and targets.

        :param str phase: The phase of the training - ['train', 'val', "test"].
        :param torch.Tensor inputs: A batch of input data.
        :param torch.Tensor inputs_mask: A boolean mask specifying which\
            elements in the input sequences are padded and should be ignored.
        :param torch.Tensor targets: A batch of target data.
        :param nn.Module loss_func: The loss function to use.
        :param Optimizer optimizer: The optimizer to use.
        :param float reg_lambda: The regularization parameter. Default: 0.0.
        :param Optional[str] reg_type: The type of regularization to use.\
            Can be ['L1', 'L2', None]. Default: None.

        :return tuple[float, torch.Tensor]: A tuple of the value\
            of the loss function and attention weights tensor.
        """
        outputs, attention_weights = self(inputs, inputs_mask)
        loss = loss_func(outputs[:, -1], targets)

        if reg_type in self._REG_NORMS:
            reg_term = sum(torch.norm(
                param, self._REG_NORMS[reg_type])
                for param in self.parameters())
            loss += reg_lambda * reg_term

        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item(), attention_weights

    def _validate_args(self, **args) -> None:
        """
        Validates the arguments based on their types. The arguments
        are compared to the '_types_validation' dictionary
        (argument name and type). Note: This method only accepts
        keyword arguments.

        :param `**args`: Dictionary containing the keyword arguments.

        :raise KeyError: Raises if a given keyword argument is not present\
            in the '_types_validation' dictionary of the class.
        :raise TypeError: Raises f a given argument has an incorrect type.
        """
        for arg, arg_value in args.items():
            try:
                true_arg_type = self._types_validation[arg]
            except KeyError:
                raise KeyError(
                    f"Argument {arg} is not present in the class variable\
                    '_types_validation'.")

            if not isinstance(arg_value, true_arg_type):
                raise TypeError(
                    f"Argument {arg} must be of type {true_arg_type},\
                    but got {type(arg_value)} instead.")

        self._args_validation(**args)

    def _args_validation(self, **args: dict[str, Any]) -> None:
        """
        Arguments validation for Transformer model.

        :params dict[str, Any] **args: Dictionary of model parameters.

        :raise ValueError: Raises when value of the given arg. is incorrect.
        """
        if args['input_size'] <= 0:
            raise ValueError(
                f"Parameter 'input_size' has to be positive integer.\
                Received input_size={args['input_size']}.")
        if args['model_size'] <= 0:
            raise ValueError(
                f"Parameter 'model_size' has to be positive integer.\
                Received model_size={args['model_size']}.")
        if args['num_layers'] <= 0:
            raise ValueError(
                f"Parameter 'num_layers' has to be positive integer.\
                Received num_layers={args['num_layers']}.")
        if args['dim_feedforward'] <= 0:
            raise ValueError(
                f"Parameter 'dim_feedforward' has to be positive integer.\
                Received dim_feedforward={args['dim_feedforward']}.")
        if args['num_heads'] <= 0:
            raise ValueError(
                f"Parameter 'num_heads' has to be positive integer.\
                Received num_heads={args['num_heads']}.")
        if args['dropout_rate'] <= 0:
            raise ValueError(
                f"Parameter 'dropout_rate' has to be positive float.\
                Received dropout_rate={args['dropout_rate']}.")
