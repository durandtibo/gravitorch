r"""This module defines some collators using padding to deal with variable size
data."""

__all__ = ["PaddedSequenceCollator", "DictPaddedSequenceCollator"]

from collections.abc import Hashable, Sequence

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from gravitorch import constants as ct
from gravitorch.data.dataloaders.collators.base import BaseCollator


class PaddedSequenceCollator(BaseCollator[tuple[dict, dict], dict]):
    r"""Implements a collate function to create zero padded sequences.

    Each sequence can have a different length. The missing "values"
    are filled with 0 (default value). The examples have to be
    formatted in a specified format. The example should be a tuple
    with two items. The input should be a tuple with two items. The
    first item is a dict containing the fixed size inputs i.e. the
    input that have the same size for each example of the mini-batch.
    The first item needs to contain the length of the sequence
    because this information is used to sort the examples in the
    batch by descending order of sequence length. The length of the
    sequence is indicated by the key ``'length'``. The first item is
    combined by the default PyTorch collate function. The second item
    is a dict containing all the inputs that have variable size. This
    item is combined by using packed sequences. This collator does
    not work if all the sequences are empty. Note that every empty
    sequence if removed.

    Args:
        length_key (hashable, optional): Specifies the key with the
            length of each example. Default: ``'length'``
        batch_first (bool, optional): Specifies if the first dimension
            is the batch size or the sequence length.
            Default: ``False``
        padding_value (float, optional): value for padded elements.
            Default: ``0.0``
    """

    def __init__(
        self,
        length_key: Hashable = ct.LENGTH,
        batch_first: bool = False,
        padding_value: float = 0.0,
    ):
        self._length_key = length_key
        self._batch_first = batch_first
        self._padding_value = padding_value

    def __call__(self, data: list[tuple[dict, dict]]) -> dict:
        r"""Creates a batch of packed sequence examples given a list of
        examples.

        Args:
            data (list): The list of examples.

        Returns:
            dict: The generated mini-batch.
        """
        # Sort the examples by the length of their sequence.
        data.sort(key=lambda x: x[0][self._length_key], reverse=True)

        fixed_size_data, variable_size_data = zip(*data)
        # Create the batch with the fixed size inputs and remove examples with empty sequences.
        batch = default_collate(fixed_size_data)
        lengths = batch[self._length_key]
        num_seq = (lengths > 0).sum().item()

        # Remove the empty sequences.
        for key, value in batch.items():
            batch[key] = value[:num_seq]

        # Create the zero padded sequences.
        for key in variable_size_data[0].keys():
            batch[key] = pad_sequence(
                [variable_size_data[i][key] for i in range(num_seq)],
                batch_first=self._batch_first,
                padding_value=self._padding_value,
            )

        return batch

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"length_key={self._length_key}, "
            f"batch_first={self._batch_first}, "
            f"padding_value={self._padding_value})"
        )


class DictPaddedSequenceCollator(BaseCollator[dict, dict]):
    r"""Implements a collator to create batch of padded sequences because the
    default PyTorch collate function does not work with sequences of different
    lengths.

    With this collator, each example in the batch can have a different
    length. Each example has to be a dict, and you need to specify the
    list of keys that will be aggregated by using the function
    ``pad_sequence`` of PyTorch.

    .. note::

        This collator will not raise an error if you specify a key
        that is not in the keys of the batch.

    Args:
        keys_to_pad (list): Specifies the list of keys to pack. The
            first key is used to sort the examples by length.
        batch_first (bool, optional): Specifies if the first dimension
            is the batch size or the sequence length.
            Default: ``False``
        padding_value (float, optional): value for padded elements.
            Default: ``0.``
    """

    def __init__(
        self,
        keys_to_pad: Sequence[Hashable],
        batch_first: bool = False,
        padding_value: float = 0.0,
    ):
        self._keys_to_pad = tuple(keys_to_pad)
        self._batch_first = batch_first
        self._padding_value = padding_value

    def __call__(self, data: Sequence[dict]) -> dict:
        r"""Creates a batch of packed sequence examples given a list of
        examples.

        Args:
            data: The sequence of examples.

        Returns:
            dict: The generated mini-batch.
        """
        # Remove the empty sequences.
        data = [example for example in data if example[self._keys_to_pad[0]].shape[0] > 0]
        num_seq = len(data)

        # Sort the examples by the length of their sequence.
        data.sort(key=lambda x: x[self._keys_to_pad[0]].shape[0], reverse=True)

        # Create the batch of examples.
        batch = {}
        for key in data[0].keys():
            examples = [data[i][key] for i in range(num_seq)]
            if key in self._keys_to_pad:
                batch[key] = pad_sequence(
                    examples, batch_first=self._batch_first, padding_value=self._padding_value
                )
            else:
                batch[key] = default_collate(examples)

        return batch

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"keys_to_pack={self._keys_to_pad}, "
            f"batch_first={self._batch_first}, "
            f"padding_value={self._padding_value})"
        )
