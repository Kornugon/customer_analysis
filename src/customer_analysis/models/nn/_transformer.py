import copy
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F


class TransformerEncoderLayer(nn.Module):
    """
    Modification based on torch/nn/modules/transformer.py (torch==1.13.1),
    in purpose of posibility to return attention weights.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 batch_first: bool = False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask,
                                       average_attn_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, weights


class TransformerEncoder(nn.Module):
    """
    Modification based on torch/nn/modules/transformer.py (torch==1.13.1),
    in purpose of posibility to return attention weights.
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask,
                                 src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(
        activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))


class PositionalEncoder(nn.Module):
    """
    The PositionalEncoder module injects information about the relative
    or absolute position of the tokens in the sequence. The positional
    encodings have the same dimension as the embeddings so that the two
    can be summed. Implemented based on:

    :ref: LANGUAGE MODELING WITH NN.TRANSFORMER AND TORCHTEXT
        <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>

    :param  intmodel_size: The dimension of the model.
    :param float dropout: The dropout rate. Default: 0.1.
    :param int max_len: The maximum length of the sequence. Default: 1000.
    """

    def __init__(
            self,
            model_size: int,
            dropout: float = 0.1,
            max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0., model_size, 2.) * -
            (torch.log(torch.tensor(10000.0)) / model_size)
        )
        pos_encod = torch.zeros(max_len, 1, model_size)
        pos_encod[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encod[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pos_encod', pos_encod)

    def forward(
            self,
            seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalEncoder.

        :param torch.Tensor x: Input tensor of shape\
            (seq_len, batch_size, embedding_dim).

        :return torch.Tensor: Sum of positional encodings and\
            the input tensor.
        """
        return self.dropout(seq + self.pos_encod[:seq.size(0), :])
