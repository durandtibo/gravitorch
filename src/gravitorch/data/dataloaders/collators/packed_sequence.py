r"""This module defines some collators that use packed sequence (https:/

/pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.htm
l) to deal with variable size data.
"""

__all__ = ["PackedSequenceCollator", "DictPackedSequenceCollator"]

from collections.abc import Hashable, Sequence

from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataloader import default_collate

from gravitorch import constants as ct
from gravitorch.data.dataloaders.collators.base import BaseCollator


class PackedSequenceCollator(BaseCollator[tuple[dict, dict], dict]):
    """Implements a collator to create batch of packed sequences because the
    default PyTorch collate function does not work with sequences of different
    lenghts.

    With this collator, each example in the batch can have a different
    length. The examples have to be formatted in a specified format.
    The example should be a tuple with two items. The first item is a
    dict containing the fixed size inputs i.e. the input that have the
    same size for each example of the mini-batch. The first item needs
    to contain the length of the sequence because this information is
    used to sort the examples in the batch by descending order of
    sequence length. The length of the sequence is indicated by the
    key ``'length'``. The first item is combined by the default
    PyTorch collate function. The second item is a dict containing all
    the inputs that have variable size. Each variable size input
    should be of size ``L x *``, where `L` is the length of a sequence
    and `*` is any number of trailing dimensions, including zero. This
    item is combined by using packed sequences. This collator does not
    work if all the sequences are empty.

    Note that every empty sequence if removed.

    Args:
        length_key (``Hashable``, optional): Specifies the key with
            the length of each example. Default: ``'length'``
    """

    def __init__(self, length_key: Hashable = ct.LENGTH):
        self._length_key = length_key

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

        # Create the packed sequences.
        for key in variable_size_data[0].keys():
            batch[key] = pack_sequence([variable_size_data[i][key] for i in range(num_seq)])

        return batch

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(length_key={self._length_key})"


class DictPackedSequenceCollator(BaseCollator[dict, dict]):
    r"""Implements a collator to create batch of packed sequences because the
    default PyTorch collate function does not work with sequences of different
    lengths.

    With this collator, each example in the batch can have a different
    length. Each example has to be a dict, and you need to specify the
    list of keys that will be aggregated by using the function
    ``pack_sequence`` of PyTorch. Each input to pack should be of size
    ``L x *``, where `L` is the length of a sequence and `*` is any
    number of trailing dimensions, including zero.

    .. note::

        This collator will not raise an error if you specify a key
        that is not in the keys of the batch.

    Args:
        keys_to_pack (``Sequence``): Specifies the sequence of keys
            to pack. The first key is used to sort the examples by
            length.
    """

    def __init__(self, keys_to_pack: Sequence[Hashable]):
        self._keys_to_pack = tuple(keys_to_pack)

    def __call__(self, data: Sequence[dict]) -> dict:
        r"""Creates a batch of packed sequence examples given a list of
        examples.

        Args:
            data (``Sequence``): The sequence of examples.

        Returns:
            dict: The generated mini-batch.
        """
        # Remove the empty sequences.
        data = [example for example in data if example[self._keys_to_pack[0]].shape[0] > 0]
        num_seq = len(data)

        # Sort the examples by the length of their sequence.
        data.sort(key=lambda x: x[self._keys_to_pack[0]].shape[0], reverse=True)

        # Create the batch of examples.
        batch = {}
        for key in data[0].keys():
            examples = [data[i][key] for i in range(num_seq)]
            if key in self._keys_to_pack:
                batch[key] = pack_sequence(examples)
            else:
                batch[key] = default_collate(examples)

        return batch

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(keys_to_pack={self._keys_to_pack})"
