from typing import Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset


class EventSequenceDataPreparation(Dataset):
    """
    Class for data preparation.
    Designed for model Transformer and RNN for churn analysis
    and next event prediction in a sequence of events.
    """

    def __init__(
            self,
            padding_value: int,
            event2idx: dict[str, int],
            input_df: pd.DataFrame,
            sequence_column: str,
            target_column: Optional[str] = None,
            model_type: str = 'rnn',
            n_last_events: int = 1000,
            include_test_targets: bool = True,
            train_size: float = 0.7,
            val_size: float = 0.15):
        """
        :param int padding_value: The value to use for padding.
        :param dict[str, int] event2idx: A dictionary that maps \
            event names to indices.
        :param pd.DataFrame input_df: An input DataFrame.
        :param str sequence_column: The name of the column in the \
            DataFrame that contains the sequences.
        :param Optional[str] target_column: The name of the column in the \
            DataFrame that contains the targets. Default: None.
        :param str model_type: The type of model to use. Must be one of \
            'rnn' or 'transformer'. Default: 'rnn'.
        :param int n_last_events: Amount of latest events to keep. \
            Default: 1000.
        :param bool include_test_targets: Whether to include the real targets \
            for the test set or not. Default: True.
        :param float train_size: The proportion of the data to use \
            for training. Default: 0.7.
        :param float val_size: The proportion of the data to use f\
            or validation. Default: 0.15.
        """
        self.padding_value = padding_value
        self.event2idx = event2idx
        self.sequence_column = sequence_column
        self.target_column = target_column
        self.model_type = model_type
        self.n_last_events = n_last_events
        self.include_test_targets = include_test_targets
        self.input_df = input_df
        self.train_size = train_size
        self.val_size = val_size

        self.datasets_dict = self._prepare_data()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        :return int: The length of the dataset.
        """
        return len(self.datasets_dict)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Returns a tuple of tensors for the given index.

        :param int idx: The index to retrieve.

        :return tuple[torch.Tensor, ...]: A tuple of tensors for the \
            given index,containing the padded sequence, the target, \
            and optionally the sequence length.
        """
        return self.datasets_dict[idx]

    def _prepare_data(self) -> dict[str, TensorDataset]:
        """
        Prepares a dictionary that maps phase names ['train', 'val', 'test']
        to TensorDatasets for each phase from the input DataFrame.

        :return dict[str, TensorDataset]: A dictionary that maps phase names \
            ['train', 'val', 'test'] to TensorDatasets for each phase.
        """
        self.input_df[self.sequence_column] = self.input_df[
            self.sequence_column].apply(lambda x: x[-self.n_last_events:])

        train_df, val_df, test_df = self._split_data(self.input_df)

        datasets_dict = {
            'train': self._create_dataset(train_df),
            'val': self._create_dataset(val_df),
            'test': self._create_dataset(test_df)
        }

        return datasets_dict

    def _split_data(
        self,
        input_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the input DataFrame into train, validation,
        and test sets according to the given proportions.

        :param pd.DataFrame input_df: An input DataFrame.

        :return tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple \
            containing DataFrames for train, validation and \
            test sets respectively.
        """
        len_data = len(input_df)
        train_size = int(self.train_size * len_data)
        val_size = int(self.val_size * len_data)

        train_df = input_df.iloc[:train_size]
        val_df = input_df.iloc[train_size:train_size + val_size]
        test_df = input_df.iloc[train_size + val_size:]

        return train_df, val_df, test_df

    def _create_dataset(
            self,
            df: pd.DataFrame) -> TensorDataset:
        """
        Creates a TensorDataset from an input DataFrame.

        :param pd.DataFrame df: An input DataFrame.

        :raises ValueError: Raises when model has any sequence\
           with just one or less events, for next event prediction.

        :return TensorDataset: A TensorDataset containing prepared data from \
            an input DataFrame.
        """
        sequences = df[self.sequence_column].tolist()
        sequences = [[self.event2idx[event]
                      for event in seq] for seq in sequences]

        if self.target_column:
            targets = df[self.target_column].tolist()
        else:
            sequences, targets = (
                [seq[:-1] for seq in sequences],
                [seq[-1] for seq in sequences])

        if not self.include_test_targets:
            targets = [torch.tensor(0) for _ in range(len(sequences))]
        else:
            targets = [torch.tensor(target) for target in targets]

        sequences = [torch.tensor(seq) for seq in sequences]

        seq_lengths = torch.tensor([len(seq) for seq in sequences])
        if any(length <= 0 for length in seq_lengths):
            sequ_txt_1 = f'Existing sequence length <= 0 in: {seq_lengths}. '
            sequ_txt_2 = f'Input sequences should have at least 2 events.'
            raise ValueError(f'{sequ_txt_1}{sequ_txt_2}')

        padded_sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.padding_value)

        if self.model_type == 'rnn':
            targets = torch.tensor(targets).to(torch.long)
            padded_sequences = padded_sequences.to(torch.float32)
            dataset = TensorDataset(
                padded_sequences.unsqueeze(-1), targets, seq_lengths)
        else:
            targets = torch.stack(targets).to(torch.long)
            padded_sequences = padded_sequences.to(torch.long)
            dataset = TensorDataset(padded_sequences, targets)

        return dataset
