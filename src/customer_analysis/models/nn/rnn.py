from abc import ABC
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence)


class RNNModel(nn.Module, ABC):
    """
    The model is used to create a Recurrent Neural Network with
    attention mechanisms to predict next event or churn,
    based on sequences of events.
    """
    _REG_NORMS = {'L1': 1, 'L2': 2}
    _types_validation = {'hidden_size': int,
                         'num_layers': int,
                         'num_classes': int,
                         'padding_value': int,
                         'nonlinearity': str,
                         'attention_type': str,
                         'num_heads': int}

    def __init__(
            self,
            hidden_size: int,
            num_layers: int,
            num_classes: int,
            device: torch.device,
            padding_value: int,
            nonlinearity: str = 'relu',
            attention_type: str = 'global',
            num_heads: int = 1) -> None:
        """
        :param int hidden_size: The number of features in the hidden state.
        :param int num_layers: Number of recurrent layers.
        :param int num_classes: Number of classes for classification.
        :param torch.device device: Selected compute device from\
            [`cpu`, `cuda`].
        :param int padding_value: The value used to pad the input sequences\
            to the same length.
        :param str nonlinearity: The non-linearity to use.\
            Can be ['tanh', 'relu']. Default: 'relu'.
        :param str attention_type: Type of attention mechanism to use.\
            Can be ['self', 'global', 'multi-head']. Default: 'global'.
        :param int num_heads: Number of heads for multi-head attention.\
            Only used, when attention_type is 'multi-head'. Default: 1.

        Input shape
            - A 3D tensor with shape (batch_size, seq_len, input_size).

        Output shape
            - A 2D tensor with shape (batch_size, num_classes).
        """
        super().__init__()
        self._validate_args(hidden_size=hidden_size,
                            num_layers=num_layers,
                            num_classes=num_classes,
                            padding_value=padding_value,
                            nonlinearity=nonlinearity,
                            attention_type=attention_type,
                            num_heads=num_heads)

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_value = padding_value
        self.attention_type = attention_type

        # in nn.RNN -> input_size = 1
        self.rnn = nn.RNN(1, hidden_size, num_layers,
                          batch_first=True, nonlinearity=nonlinearity)
        self.fc = nn.Linear(hidden_size, num_classes)

        attention_dict = {
            'self': nn.Linear(hidden_size * 2, 1),
            'global': nn.Linear(hidden_size, 1),
            'multi-head': nn.ModuleList(
                [nn.Linear(hidden_size, 1) for _ in range(num_heads)])
        }
        self.attention = attention_dict.get(attention_type, 'global')
        self.to(self.device)

    def forward(
            self,
            seq: torch.Tensor,
            seq_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param torch.Tensor seq: The input sequence data.
        :param torch.Tensor seq_lengths: A tensor of integers representing the\
            lengths of the input sequences, excluding any padded values.

        :return tuple[torch.Tensor, torch.Tensor]: Tuple of the model outputs\
            and attention weights.
        """
        # Pack the padded input sequence
        x = pack_padded_sequence(
            seq, seq_lengths, batch_first=True, enforce_sorted=False)

        # Pass the packed sequence through the RNN layer
        hidden = torch.zeros(
            self.num_layers, x.batch_sizes[0], self.hidden_size)\
            .to(self.device)
        out, _ = self.rnn(x, hidden)

        # Unpack the output sequence
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     padding_value=self.padding_value)

        # Create a mask to exclude padded values from the attention mechanism
        inputs_mask = (out != self.padding_value).to(self.device)
        # Apply the attention mechanism
        attention_func_dict = {
            'self': self._attention_self,
            'global': self._attention_global,
            'multi-head': self._attention_multi_head
        }
        attention_func = attention_func_dict.get(self.attention_type)
        out, attention_weights = attention_func(out, inputs_mask)
        out = self.fc(out)

        return out, attention_weights

    def predict(
            self,
            sequence: torch.Tensor,
            seq_lengths: torch.Tensor,
            top_k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the top-k predictions for the next event in a sequence
        of events or churn based on sequence of events.

        :param torch.Tensor sequence: The input data.
        :param torch.Tensor seq_lengths: A tensor of integers representing the\
            lengths of the input sequences, excluding any padded values.
        :param int top_k: The number of top results to return. Default: 1.

        :return tuple[torch.Tensor, torch.Tensor]: A tensor of shape\
            (batch_size, k) containing the indices of the top-k predicted\
            events or churns for each input sequence and attention weights.
        """
        output, attention_weights = self(sequence, seq_lengths)
        _, top_k_indices = torch.topk(output, top_k, dim=-1)

        return top_k_indices, attention_weights

    def predict_prob(
            self,
            sequence: torch.Tensor,
            seq_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the probabilities for each class for the churn
        based on sequence of events.

        :param torch.Tensor sequence: The input data.
        :param torch.Tensor seq_lengths: A tensor of integers representing the\
            lengths of the input sequences, excluding any padded values.

        :return tuple[torch.Tensor, torch.Tensor]: A tensor of shape\
            (batch_size, num_classes) containing the probabilities\
            for each class for each input sequence and attention weights.
        """
        output, attention_weights = self(sequence, seq_lengths)
        probabilities = torch.softmax(output, dim=-1)

        return probabilities, attention_weights

    def step(
            self,
            phase: str,
            inputs: torch.Tensor,
            seq_lengths: torch.Tensor,
            targets: torch.Tensor,
            loss_func: nn.Module,
            optimizer: Optimizer,
            reg_lambda: float = 0.0,
            reg_type: Optional[str] = None) -> tuple[float, torch.Tensor]:
        """
        Performs one step of training on the given inputs and targets.

        :param str phase: The phase of the training - ['train', 'val', "test"].
        :param torch.Tensor inputs: The input data.
        :param torch.Tensor seq_lengths: A tensor representing the\
            lengths of the input sequences, excluding any padded values.
        :param torch.Tensor targets: The target data.
        :param nn.Module loss_func: The loss function to use.
        :param Optimizer optimizer: The optimizer to use.
        :param float reg_lambda: The regularization parameter. Default: 0.0.
        :param Optional[str] reg_type: The type of regularization to use.\
            Can be ['L1', 'L2'] or None. Default: None.

        :return tuple[float, torch.Tensor]: A tuple of the value\
            of the loss function and attention weights tensor.
        """
        outputs, attention_weights = self(inputs, seq_lengths)
        loss = loss_func(outputs, targets)

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

    def _attention_self(
            self,
            out: torch.Tensor,
            mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Self-attention mechanism: This mechanism computes the
        interactions between all pairs of output vectors and
        uses these interactions to compute the attention weights.
        This allows the model to focus on certain parts of the
        input sequence when making its prediction.

        :param torch.Tensor out: The output tensor from the RNN layer.\
            Shape: (batch_size, seq_len, hidden_size).
        :param torch.Tensor mask: The mask tensor.\
            Shape: (batch_size, seq_len).

        :return tuple[torch.Tensor, torch.Tensor]: A tuple containing tensors:
            - output: Output tensor after applying the attention mechanism.\
              Shape: (batch_size, hidden_size).
            - weights: Attention weights for each head.\
              Shape: (batch_size, seq_len).
        """
        # Learn a context vector to compute the similarity between out vectors
        context_vector = nn.Parameter(torch.randn(out.shape[-1]))\
            .to(self.device)
        attention_weights = torch.matmul(out, context_vector)
        attention_weights.masked_fill_(~mask.any(dim=-1), -float('inf'))
        attention_weights = nn.functional.softmax(attention_weights, dim=-1)

        return torch.sum(out * attention_weights.unsqueeze(-1), dim=1), \
            attention_weights

    def _attention_global(
            self,
            out: torch.Tensor,
            mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Global attention mechanism: This mechanism computes the
        attention weights based on the individual output vectors.
        This allows the model to focus on certain parts of the
        input sequence when making its prediction.

        :param torch.Tensor out: The output tensor from the RNN layer.\
            Shape: (batch_size, seq_len, hidden_size).
        :param torch.Tensor mask: The mask tensor.\
            Shape: (batch_size, seq_len).

        :return tuple[torch.Tensor, torch.Tensor]: A tuple containing tensors:
            - output: Output tensor after applying the attention mechanism.\
              Shape: (batch_size, hidden_size).
            - weights: Attention weights for each head.\
              Shape: (batch_size, seq_len).
        """
        attention_weights = self.attention(out)
        attention_weights = attention_weights.squeeze(-1)
        attention_weights.masked_fill_(~mask.any(dim=-1), -float('inf'))
        attention_weights = nn.functional.softmax(attention_weights, dim=1)

        return torch.sum(out * attention_weights.unsqueeze(-1), dim=1), \
            attention_weights

    def _attention_multi_head(
            self,
            out: torch.Tensor,
            mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention mechanism: This mechanism computes
        multiple sets of attention weights using multiple heads
        and combines their results. This allows the model to focus
        on different parts of the input sequence simultaneously
        when making its prediction. This function uses the _attention_self()
        function per head to compute the attention weights.

        :param torch.Tensor out: The output tensor from the RNN layer.\
            Shape: (batch_size, seq_len, hidden_size).
        :param torch.Tensor mask: The mask tensor.\
            Shape: (batch_size, seq_len).

        :return tuple[torch.Tensor, torch.Tensor]: A tuple containing tensors:
            - output: Output tensor after applying the attention mechanism.\
              Shape: (batch_size, hidden_size).
            - weights: Attention weights for each head.\
              Shape: (num_heads, batch_size, seq_len).
        """
        attention_weights = []
        for _ in self.attention:
            _, head_weights = self._attention_self(out, mask)
            attention_weights.append(head_weights)

        out = torch.stack([torch.sum(out * att_w.unsqueeze(-1), dim=1)
                           for att_w in attention_weights], dim=-1)

        return torch.mean(out, dim=-1), torch.stack(attention_weights, dim=0)

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
        Arguments validation for RNN model.

        :params dict[str, Any] **args: Dictionary of model parameters.

        :raise ValueError: Raises when value of the given arg. is incorrect.
        """
        nonlinearity_options = ('tanh', 'relu')
        attention_type_options = ('self', 'global', 'multi-head')

        if args['hidden_size'] <= 0:
            raise ValueError(
                f"Parameter 'hidden_size' has to be positive integer.\
                Received hidden_size={args['hidden_size']}.")
        if args['num_layers'] <= 0:
            raise ValueError(
                f"Parameter 'num_layers' has to be positive integer.\
                Received num_layers={args['num_layers']}.")
        if args['num_classes'] <= 0:
            raise ValueError(
                f"Parameter 'num_classes' has to be positive integer.\
                Received num_classes={args['num_classes']}.")
        if args['num_heads'] <= 0:
            raise ValueError(
                f"Parameter 'num_heads' has to be positive integer.\
                Received num_heads={args['num_heads']}.")
        if args['padding_value'] <= 0:
            raise ValueError(
                f"Parameter 'padding_value' has to be positive integer.\
                Received padding_value={args['padding_value']}.")
        if args['nonlinearity'] not in nonlinearity_options:
            raise ValueError(
                f"Parameter 'nonlinearity' has to be one of: \
                {nonlinearity_options}. \
                Received nonlinearity={args['nonlinearity']}.")
        if args['attention_type'] not in attention_type_options:
            raise ValueError(
                f"Parameter 'attention_type' has to be one of: \
                {attention_type_options}. \
                Received attention_type={args['attention_type']}.")
