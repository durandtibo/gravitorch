from gravitorch.experimental.batchloader import IterableBatchLoader

#########################################
#     Tests for IterableBatchLoader     #
#########################################


def test_iterable_batchloader_str() -> None:
    assert str(IterableBatchLoader([1, 2, 3, 4, 5])).startswith("IterableBatchLoader(")


def test_iterable_batchloader_iter() -> None:
    assert tuple(IterableBatchLoader([1, 2, 3, 4, 5])) == (1, 2, 3, 4, 5)


def test_iterable_batchloader_with_iter() -> None:
    with IterableBatchLoader([1, 2, 3, 4, 5]) as loader:
        assert tuple(loader) == (1, 2, 3, 4, 5)
