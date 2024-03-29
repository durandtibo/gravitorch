from typing import Union

import torch
from objectory import OBJECT_TARGET
from pytest import mark, raises

from gravitorch.utils.format import (
    human_byte_size,
    human_count,
    human_time,
    str_mapping,
    str_pretty_dict,
    str_pretty_json,
    str_pretty_yaml,
    str_scalar,
    str_target_object,
)

#####################################
#     Tests for human_byte_size     #
#####################################


@mark.parametrize("size,output", ((2, "2.00 B"), (2048, "2,048.00 B"), (2097152, "2,097,152.00 B")))
def test_human_byte_size_b(size: int, output: str) -> None:
    assert human_byte_size(size, "B") == output


@mark.parametrize(
    "size,output",
    [(2048, "2.00 KB"), (2097152, "2,048.00 KB"), (2147483648, "2,097,152.00 KB")],
)
def test_human_byte_size_kb(size: int, output: str) -> None:
    assert human_byte_size(size, "KB") == output


@mark.parametrize(
    "size,output",
    [(2048, "0.00 MB"), (2097152, "2.00 MB"), (2147483648, "2,048.00 MB")],
)
def test_human_byte_size_mb(size: int, output: str) -> None:
    assert human_byte_size(size, "MB") == output


@mark.parametrize("size,output", [(2048, "0.00 GB"), (2097152, "0.00 GB"), (2147483648, "2.00 GB")])
def test_human_byte_size_gb(size: int, output: str) -> None:
    assert human_byte_size(size, "GB") == output


@mark.parametrize(
    "size,output",
    [(2, "2.00 B"), (1023, "1,023.00 B"), (2048, "2.00 KB"), (2097152, "2.00 MB")],
)
def test_human_byte_size_auto(size: int, output: str) -> None:
    assert human_byte_size(size) == output


def test_human_byte_size_incorrect_unit() -> None:
    with raises(ValueError, match="Incorrect unit ''. The available units are"):
        assert human_byte_size(1, "")


#################################
#     Tests for human_count     #
#################################


@mark.parametrize(
    "count,human",
    [
        (0, "0"),
        (123, "123"),
        (123.5, "123"),
        (1234, "1.2 K"),
        (2e6, "2.0 M"),
        (3e9, "3.0 B"),
        (4e14, "400 T"),
        (5e15, "5,000 T"),
    ],
)
def test_human_count(count: Union[int, float], human: str) -> None:
    assert human_count(count) == human


def test_human_count_incorrect_value() -> None:
    with raises(ValueError, match="The number should be a positive number"):
        human_count(-1)


################################
#     Tests for human_time     #
################################


@mark.parametrize(
    "seconds,human",
    (
        (1, "0:00:01"),
        (61, "0:01:01"),
        (3661, "1:01:01"),
        (3661.0, "1:01:01"),
        (1.1, "0:00:01.100000"),
        (3600 * 24 + 3661, "1 day, 1:01:01"),
        (3600 * 48 + 3661, "2 days, 1:01:01"),
    ),
)
def test_human_time(seconds: Union[int, float], human: str) -> None:
    assert human_time(seconds) == human


#################################
#     Tests for str_mapping     #
#################################


def test_str_mapping_empty() -> None:
    assert str_mapping({}) == ""


def test_str_mapping_one() -> None:
    assert str_mapping({"key1": "value1"}) == "key1=value1"


def test_str_mapping_two() -> None:
    assert str_mapping({"key1": "value1", "key2": "value2"}) == "key1=value1\nkey2=value2"


def test_str_mapping_multilines() -> None:
    assert (
        str_mapping({"key1": "long\nvalue1", "key2": "value2"})
        == "key1=long\n  value1\nkey2=value2"
    )


def test_str_mapping_sorted_keys_false() -> None:
    assert str_mapping({"key2": "value2", "key1": "value1"}) == "key2=value2\nkey1=value1"


def test_str_mapping_sorted_keys_true() -> None:
    assert (
        str_mapping({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "key1=value1\nkey2=value2"
    )


def test_str_mapping_num_spaces_4() -> None:
    assert (
        str_mapping({"key1": "long\nvalue1", "key2": "value2"}, num_spaces=4)
        == "key1=long\n    value1\nkey2=value2"
    )


def test_str_mapping_one_line_empty() -> None:
    assert str_mapping({}, one_line=True) == ""


def test_str_mapping_one_line_one() -> None:
    assert str_mapping({"key1": "value1"}, one_line=True) == "key1=value1"


def test_str_mapping_one_line_two() -> None:
    assert (
        str_mapping({"key1": "value1", "key2": "value2"}, one_line=True)
        == "key1=value1, key2=value2"
    )


################################
#     Tests for str_scalar     #
################################


def test_str_scalar_small_positive_int_value() -> None:
    assert str_scalar(1234567) == "1,234,567"


def test_str_scalar_small_negative_int_value() -> None:
    assert str_scalar(-1234567) == "-1,234,567"


def test_str_scalar_large_positive_int_value() -> None:
    assert str_scalar(12345678901) == "1.234568e+10"


def test_str_scalar_large_negative_int_value() -> None:
    assert str_scalar(-12345678901) == "-1.234568e+10"


def test_str_scalar_positive_float_value_1() -> None:
    assert str_scalar(123456.789) == "123,456.789000"


def test_str_scalar_positive_float_value_2() -> None:
    assert str_scalar(0.123456789) == "0.123457"


def test_str_scalar_negative_float_value_1() -> None:
    assert str_scalar(-123456.789) == "-123,456.789000"


def test_str_scalar_negative_float_value_2() -> None:
    assert str_scalar(-0.123456789) == "-0.123457"


def test_str_scalar_small_positive_float_value() -> None:
    assert str_scalar(9.123456789e-4) == "9.123457e-04"


def test_str_scalar_small_negative_float_value() -> None:
    assert str_scalar(-9.123456789e-4) == "-9.123457e-04"


def test_str_scalar_large_positive_float_value() -> None:
    assert str_scalar(9.123456789e8) == "9.123457e+08"


def test_str_scalar_large_negative_float_value() -> None:
    assert str_scalar(-9.123456789e8) == "-9.123457e+08"


#######################################
#     Tests for str_target_object     #
#######################################


def test_str_target_object_with_target() -> None:
    assert (
        str_target_object({OBJECT_TARGET: "something.MyClass"}) == "[_target_: something.MyClass]"
    )


def test_str_target_object_without_target() -> None:
    assert str_target_object({}) == "[_target_: N/A]"


#####################################
#     Tests for str_pretty_json     #
#####################################


def test_str_pretty_json_small_dict() -> None:
    assert str_pretty_json(data={"my_key": "my_value"}) == "{'my_key': 'my_value'}"


def test_str_pretty_json_small_list() -> None:
    assert str_pretty_json(data=["my_key", "my_value"]) == "['my_key', 'my_value']"


def test_str_pretty_json_small_dict_max_len_0() -> None:
    assert str_pretty_json(data={"my_key": "my_value"}, max_len=0) == '{\n  "my_key": "my_value"\n}'


def test_str_pretty_json_small_list_max_len_0() -> None:
    assert (
        str_pretty_json(data=["my_key", "my_value"], max_len=0) == '[\n  "my_key",\n  "my_value"\n]'
    )


def test_str_pretty_json_from_str() -> None:
    assert str_pretty_json(data="abc") == "abc"


def test_str_pretty_json_from_tensor() -> None:
    assert (
        str_pretty_json(data=torch.zeros(2, 3)) == "tensor([[0., 0., 0.],\n        [0., 0., 0.]])"
    )


#####################################
#     Tests for str_pretty_yaml     #
#####################################


def test_str_pretty_yaml_small_dict() -> None:
    assert str_pretty_yaml(data={"my_key": "my_value"}) == "{'my_key': 'my_value'}"


def test_str_pretty_yaml_small_list() -> None:
    assert str_pretty_yaml(data=["my_key", "my_value"]) == "['my_key', 'my_value']"


def test_str_pretty_yaml_small_dict_max_len_0() -> None:
    assert (
        str_pretty_yaml(data={"my_key1": "my_value1", "my_key2": "my_value2"}, max_len=0)
        == "my_key1: my_value1\nmy_key2: my_value2\n"
    )


def test_str_pretty_yaml_small_list_max_len_0() -> None:
    assert str_pretty_yaml(data=["my_key", "my_value"], max_len=0) == "- my_key\n- my_value\n"


def test_str_pretty_yaml_from_str() -> None:
    assert str_pretty_yaml(data="abc") == "abc"


def test_str_pretty_yaml_from_tensor() -> None:
    assert (
        str_pretty_yaml(data=torch.zeros(2, 3)) == "tensor([[0., 0., 0.],\n        [0., 0., 0.]])"
    )


#####################################
#     Tests for str_pretty_dict     #
#####################################


def test_str_pretty_dict_empty() -> None:
    assert str_pretty_dict(data={}) == ""


def test_str_pretty_dict_1_value() -> None:
    assert str_pretty_dict(data={"my_key": "my_value"}) == "my_key : my_value"


def test_str_pretty_dict_sorted_key() -> None:
    assert (
        str_pretty_dict(data={"short": 123, "long_key": 2}, sorted_keys=True)
        == "long_key : 2\nshort    : 123"
    )


def test_str_pretty_dict_unsorted_key() -> None:
    assert (
        str_pretty_dict(data={"short": 123, "long_key": 2}, sorted_keys=False)
        == "short    : 123\nlong_key : 2"
    )


def test_str_pretty_dict_nested_dict() -> None:
    assert str_pretty_dict(data={"my_key": {"my_key2": 123}}) == "my_key : {'my_key2': 123}"


def test_str_pretty_dict_incorrect_indent() -> None:
    with raises(ValueError, match="The indent has to be greater or equal to 0"):
        str_pretty_dict(data={"my_key": "my_value"}, indent=-1)


def test_str_pretty_dict_indent_2() -> None:
    assert str_pretty_dict(data={"my_key": "my_value"}, indent=2) == "  my_key : my_value"
