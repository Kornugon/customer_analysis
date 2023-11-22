import os
import json
from typing import Optional, Any
from itertools import accumulate, dropwhile

import pandas as pd


def read_rnn_attention_data(
        file_path: Optional[str] = None,
        phase: str = 'train',
        head_index: Optional[int] = None,
        batch_index: int = 0) -> dict[str, Any]:
    """
    Reads RNN attention data from a JSON file.
    The data schema is as follows:
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

    :param Optional[str] file_path: The path to the JSON file.\
        If not given, the best model path in current run will be used.
    :param str phase: The phase to read the data for.
    :param Optional[int] head_index: The index of the head,\
        to read the data for. Can be negative to index from the end.
    :param int batch_index: The index of the batch to read the data for.\
        Can be negative to index from the end.

    :return dict[str, Any]: A dictionary containing the attention data\
        for the specified batch.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"The file '{file_path}' does not exist.")
    if phase not in ['train', 'val', 'test']:
        raise ValueError(
            f"Invalid phase '{phase}'.\
            Phase must be one of ['train', 'val', 'test'].")

    with open(file_path, 'r') as f:
        first_batch = json.loads(f.readline())
        if first_batch.get('attention_type') != 'multi-head':
            head_index = 0
        f.seek(0)

        batch_data, phase_found = None, False
        max_head_index, max_batch_index = -1, -1
        all_batches = []
        for line in f:
            current_batch = json.loads(line)
            if current_batch['phase'] == phase:
                phase_found = True
                current_head_keys = [
                    int(key) for key in current_batch['heads'].keys()]
                current_head_index = max(current_head_keys)
                if current_head_index > max_head_index:
                    max_head_index = current_head_index
                if head_index < 0:
                    head_index = max_head_index + head_index + 1
                if head_index < 0 or head_index > max_head_index:
                    raise IndexError(
                        f"Head index {head_index} is out of range.\
                        For 'phase': '{phase}', maximum available\
                        'head_index' is {max_head_index}.")
                current_batch_index = current_batch[
                    'heads'][f'{head_index}']['batch_index']
                if current_batch_index > max_batch_index:
                    max_batch_index = current_batch_index
                all_batches.append(current_batch)

        if not phase_found:
            raise ValueError(f"The file has no data for phase '{phase}'.")

        if batch_index < 0:
            batch_index = len(all_batches) + batch_index
        if batch_index < 0 or batch_index >= len(all_batches):
            raise IndexError(
                f"Batch index {batch_index} is out of range.\
                For 'phase': '{phase}' and 'head_index': {head_index},\
                maximum available 'batch_index' is {len(all_batches)-1}.")

        batch_data = all_batches[batch_index]

        return {
            'phase': batch_data['phase'],
            'attention_type': batch_data.get('attention_type', None),
            'heads': {f'{head_index}': batch_data['heads'][f'{head_index}']}}


def read_transformer_attention_data(
        file_path: Optional[str] = None,
        phase: str = 'train',
        head_index: Optional[int] = None,
        batch_index: int = 0) -> dict[str, Any]:
    """
    Reads Transformer attention data from a JSON file.
    The data schema is as follows:
        {
            "phase": str,
            "heads": {
                head_index: {
                    "batch_index": int,
                    "batch_inputs": list[list[int]],
                    "attention_weights": list[list[float]],
                    "predictions": list[int]
                    }
                }
            }

    :param Optional[str] file_path: The path to the JSON file.\
        If not given, the best model path in current run will be used.
    :param str phase: The phase to read the data for.
    :param Optional[int] head_index: The index of the head,\
        to read the data for. Can be negative to index from the end.
    :param int batch_index: The index of the batch to read the data for.\
        Can be negative to index from the end.

    :raises FileNotFoundError: Raises when file does not exist.
    :raises ValueError: Raises when phase is invalid.
    :raises ValueError: Raises when file has no data for given phase.
    :raises IndexError: Raises when 'head_index' is out of range\
        for given phase.
    :raises IndexError: Raises when 'batch_index' is out of range\
        for given phase and 'head_index'.

    :return dict[str, Any]: A dictionary containing the attention data\
        for the specified batch.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"The file '{file_path}' does not exist.")
    if phase not in ['train', 'val', 'test']:
        raise ValueError(
            f"Invalid phase '{phase}'.\
            Phase must be one of ['train', 'val', 'test'].")

    with open(file_path, 'r') as f:
        batch_data, phase_found = None, False
        max_head_index, max_batch_index = -1, -1
        all_batches = []
        for line in f:
            current_batch = json.loads(line)
            if current_batch['phase'] == phase:
                phase_found = True
                current_head_keys = [
                    int(key) for key in current_batch['heads'].keys()]
                current_head_index = max(current_head_keys)
                if current_head_index > max_head_index:
                    max_head_index = current_head_index
                if head_index < 0:
                    head_index = max_head_index + head_index + 1
                if head_index < 0 or head_index > max_head_index:
                    raise IndexError(
                        f"Head index {head_index} is out of range.\
                        For 'phase': '{phase}', maximum available\
                        'head_index' is {max_head_index}.")
                current_batch_index = current_batch[
                    'heads'][f'{head_index}']['batch_index']
                if current_batch_index > max_batch_index:
                    max_batch_index = current_batch_index
                all_batches.append(current_batch)

        if not phase_found:
            raise ValueError(f"The file has no data for phase '{phase}'.")

        if batch_index < 0:
            batch_index = len(all_batches) + batch_index
        if batch_index < 0 or batch_index >= len(all_batches):
            raise IndexError(
                f"Batch index {batch_index} is out of range.\
                For 'phase': '{phase}' and 'head_index': {head_index},\
                maximum available 'batch_index' is {len(all_batches)-1}.")

        batch_data = all_batches[batch_index]

    return {
        'phase': batch_data['phase'],
        'heads': {f'{head_index}': batch_data['heads'][f'{head_index}']}}


def get_sequences_above_weights_threshold(
        head_data: dict[str, Any],
        idx2event: dict[int, str],
        weights_threshold: float = 0.55,
        min_events: int = 2
) -> pd.DataFrame:
    """
    This function returns a pandas DataFrame where each row corresponds
    to a sequence of events where the sum of attention weights is greater
    than or equal to the specified threshold.

    :param dict[str, Any] head_data: The data for a specific head\
        loaded by 'read_rnn_attention_data'.
    :param dict[int, str] idx2event: A dictionary mapping integer event values\
        to their string representations.
    :param float weights_threshold: The threshold for the sum of\
        attention weights. Default: 0.55.
    :param int min_events: The minimum number of events for a sequence\
        to be included. Default: 2.

    :return: A pandas DataFrame where each row corresponds to a sequence\
        of events.
    """
    def _get_sequences(
            sequence: list[int],
            weights: list[float]
    ) -> list[list[str]]:
        """
        This function returns a list of sequences where each sequence
        is a list of events and the sum of attention weights for each
        sequence is greater than or equal to the specified threshold.

        :param list[int] sequence: A list of integer event values.
        :param list[float] weights: A list of attention weights corresponding\
            to the events in the sequence.

        :return list[list[str]]: A list of sequences where each sequence\
            is a list of string event representations.
        """
        i, sequences = 0, []
        while i < len(sequence):
            cum_weights = list(
                dropwhile(lambda x: x[1] < weights_threshold, enumerate(
                    accumulate(weights[i:]), start=i)))
            if cum_weights:
                j = cum_weights[0][0] + 1
                if j - i >= min_events:
                    sequence_str = [idx2event[event]
                                    for event in sequence[i:j]]
                    sequences.append(sequence_str)
                i = j
            else:
                break
        return sequences

    sequences = [(seq_num, seq) for seq_num, (sequence, weights) in enumerate(
        zip(head_data['batch_inputs'], head_data['attention_weights']))
        for seq in _get_sequences(sequence, weights)]

    df_sequences = pd.DataFrame(
        sequences, columns=['sequence_number', 'most_relevant_events'])\
        .set_index('sequence_number')

    print(f"The produced DataFrame contains first events occurring one after "
          f"another, where the sum of the events is greater than or equal "
          f"to {round(weights_threshold * 100, 2)}%")

    return df_sequences
