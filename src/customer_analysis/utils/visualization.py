from typing import Any

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_rnn_attention_heatmap(
        heads_data: dict[str, Any],
        padding_value: int,
        idx2pred: dict[int, str]) -> None:
    """
    Function designed specifically for RNN attention data.
    Plots a heatmap of input sequences and their events importance weights.

    :param dict[str, Any] heads_data: Data for each head in the batch.
    :param int padding_value: The value used to pad the input sequences.
    :param dict[int, str] idx2even: Dictionary of events mappings.
    """
    attention_type = heads_data['attention_type']
    heads_data = heads_data['heads'][list(heads_data['heads'].keys())[0]]
    translated_preds = [idx2pred[p]
                        for p in heads_data['predictions']
                        if p != padding_value]

    data = []
    max_len = 0

    for i in range(len(heads_data['batch_inputs'])):
        inputs = heads_data['batch_inputs'][i]
        attention_weights = heads_data['attention_weights'][i]

        max_len = max(max_len, len(inputs))
        data.append(np.vstack((inputs, attention_weights)).T)

    # Pad arrays
    for i in range(len(data)):
        pad_len = max_len - len(data[i])
        data[i] = np.pad(data[i], ((0, pad_len), (0, 0)),
                         mode='constant', constant_values=np.nan)

    data = np.concatenate(data, axis=1)
    plt.rcParams['legend.fontsize'] = 12
    _, ax1 = plt.subplots(figsize=(23, 13))

    sns.heatmap(data=data[:, 1::2].T,
                cmap='YlOrRd',
                annot=data[:, ::2].astype(int).T,
                fmt='d',
                xticklabels=range(1, max_len + 1),
                ax=ax1,
                linewidths=0.5,
                cbar_kws={'location': 'right',
                          'pad': 0.073})

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels(ax1.get_yticklabels())
    ax2.set_yticklabels(translated_preds)

    ax1_title = 'Heat Map of Input Sequence and Importance Weights. '
    ax1_title += f'Atttention "{attention_type}".'
    ax1.set_title(ax1_title, fontsize=17)
    ax1.set_xlabel('Sequence Step', fontsize=15)
    ax1.set_ylabel('Input Sequence', fontsize=15)
    ax2.set_ylabel('Predictions', fontsize=15)
    ax1.figure.axes[-2].set_ylabel('Importance Weights', size=15)

    legend_dict = idx2pred
    if padding_value in legend_dict:
        legend_dict.pop(padding_value)

    legend_elements = []
    for key, value in legend_dict.items():
        legend_elements.append(plt.Line2D(
            [0], [0], color='w', label=f'{key}: {value}'))

    ax1.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(1.38, 0.5), title='Sequence Events Mapping:')

    plt.show()


def plot_transformer_attention_heatmap(
        heads_data: dict[str, Any],
        padding_value: int,
        idx2pred: dict[int, str]) -> None:
    """
    Function designed specifically for Transformer attention data.
    Plots a heatmap of input sequences and their events importance weights.

    :param dict[str, Any] heads_data: Data for each head in the batch.
    :param int padding_value: The value used to pad the input sequences.
    :param dict[int, str] idx2even: Dictionary of events mappings.
    """
    heads_data = heads_data['heads'][list(heads_data['heads'].keys())[0]]
    translated_preds = [idx2pred[p]
                        for p in heads_data['predictions']
                        if p != padding_value]

    data = []
    max_len = 0

    for i in range(len(heads_data['batch_inputs'])):
        inputs = heads_data['batch_inputs'][i]
        attention_weights = heads_data['attention_weights'][i]

        max_len = max(max_len, len(inputs))
        data.append(np.vstack((inputs, attention_weights)).T)

    # Pad arrays
    for i in range(len(data)):
        pad_len = max_len - len(data[i])
        data[i] = np.pad(data[i], ((0, pad_len), (0, 0)),
                         mode='constant', constant_values=np.nan)

    data = np.concatenate(data, axis=1)
    plt.rcParams['legend.fontsize'] = 12
    _, ax1 = plt.subplots(figsize=(23, 13))

    sns.heatmap(data=data[:, 1::2].T,
                cmap='YlOrRd',
                annot=data[:, ::2].astype(int).T,
                fmt='d',
                xticklabels=range(1, max_len + 1),
                ax=ax1,
                linewidths=0.5,
                cbar_kws={'location': 'right',
                          'pad': 0.073})

    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels(ax1.get_yticklabels())
    ax2.set_yticklabels(translated_preds)

    ax1.set_title(
        f'Heat Map of Input Sequence and Importance Weights.',
        fontsize=17)
    ax1.set_xlabel('Sequence Step', fontsize=15)
    ax1.set_ylabel('Input Sequence', fontsize=15)
    ax2.set_ylabel('Predictions', fontsize=15)
    ax1.figure.axes[-2].set_ylabel('Importance Weights', size=15)

    legend_dict = idx2pred
    if padding_value in legend_dict:
        legend_dict.pop(padding_value)

    legend_elements = []
    for key, value in legend_dict.items():
        legend_elements.append(plt.Line2D(
            [0], [0], color='w', label=f'{key}: {value}'))

    ax1.legend(handles=legend_elements, loc='center right',
               bbox_to_anchor=(1.38, 0.5), title='Sequence Events Mapping:')

    plt.show()
