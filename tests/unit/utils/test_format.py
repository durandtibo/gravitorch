from typing import Union

import numpy as np
import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET
from pytest import mark, raises

from gravitorch.utils.format import (
    convert_human_readable_count,
    convert_seconds_to_readable_format,
    str_add_indent,
    str_scalar,
    str_target_object,
    to_flat_dict,
    to_human_readable_byte_size,
    to_pretty_dict_str,
    to_pretty_json_str,
    to_pretty_yaml_str,
    to_torch_mapping_str,
    to_torch_sequence_str,
)


@mark.parametrize(
    "time_float,time_str",
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
def test_convert_seconds_to_readable_format(time_float: Union[int, float], time_str: str):
    assert convert_seconds_to_readable_format(time_float) == time_str


####################################
#     Tests for str_add_indent     #
####################################


def test_str_add_indent_1_line():
    assert str_add_indent("abc") == "abc"


def test_str_add_indent_2_lines():
    assert str_add_indent("abc\n  def") == "abc\n    def"


def test_str_add_indent_num_spaces_2():
    assert str_add_indent("abc\ndef", num_spaces=2) == "abc\n  def"


def test_str_add_indent_num_spaces_4():
    assert str_add_indent("abc\ndef", num_spaces=4) == "abc\n    def"


def test_str_add_indent_not_a_string():
    assert str_add_indent(123) == "123"


################################
#     Tests for str_scalar     #
################################


def test_str_scalar_small_positive_int_value():
    assert str_scalar(1234567) == "1,234,567"


def test_str_scalar_small_negative_int_value():
    assert str_scalar(-1234567) == "-1,234,567"


def test_str_scalar_large_positive_int_value():
    assert str_scalar(12345678901) == "1.234568e+10"


def test_str_scalar_large_negative_int_value():
    assert str_scalar(-12345678901) == "-1.234568e+10"


def test_str_scalar_positive_float_value_1():
    assert str_scalar(123456.789) == "123,456.789000"


def test_str_scalar_positive_float_value_2():
    assert str_scalar(0.123456789) == "0.123457"


def test_str_scalar_negative_float_value_1():
    assert str_scalar(-123456.789) == "-123,456.789000"


def test_str_scalar_negative_float_value_2():
    assert str_scalar(-0.123456789) == "-0.123457"


def test_str_scalar_small_positive_float_value():
    assert str_scalar(9.123456789e-4) == "9.123457e-04"


def test_str_scalar_small_negative_float_value():
    assert str_scalar(-9.123456789e-4) == "-9.123457e-04"


def test_str_scalar_large_positive_float_value():
    assert str_scalar(9.123456789e8) == "9.123457e+08"


def test_str_scalar_large_negative_float_value():
    assert str_scalar(-9.123456789e8) == "-9.123457e+08"


#######################################
#     Tests for str_target_object     #
#######################################


def test_str_target_object_with_target():
    assert (
        str_target_object({OBJECT_TARGET: "something.MyClass"}) == "[_target_: something.MyClass]"
    )


def test_str_target_object_without_target():
    assert str_target_object({}) == "[_target_: N/A]"


##################################################
#     Tests for convert_human_readable_count     #
##################################################


@mark.parametrize(
    "count,human",
    [
        (0, "0  "),
        (123, "123  "),
        (123.5, "123  "),
        (1234, "1.2 K"),
        (2e6, "2.0 M"),
        (3e9, "3.0 B"),
        (4e14, "400 T"),
        (5e15, "5,000 T"),
    ],
)
def test_convert_human_readable_count(count: Union[int, float], human: str):
    assert convert_human_readable_count(count) == human


def test_convert_human_readable_count_incorrect_value():
    with raises(ValueError):
        convert_human_readable_count(-1)


##################################
#     Tests for to_flat_dict     #
##################################


def test_to_flat_dict_flat_dict():
    flatten_data = to_flat_dict(
        {
            "bool": False,
            "float": 3.5,
            "int": 2,
            "str": "abc",
        }
    )
    assert flatten_data == {
        "bool": False,
        "float": 3.5,
        "int": 2,
        "str": "abc",
    }


def test_to_flat_dict_nested_dict_str():
    flatten_data = to_flat_dict({"a": "a", "b": {"c": "c"}, "d": {"e": {"f": "f"}}})
    assert flatten_data == {"a": "a", "b.c": "c", "d.e.f": "f"}


def test_to_flat_dict_nested_dict_multiple_types():
    flatten_data = to_flat_dict(
        {
            "module": {
                "bool": False,
                "float": 3.5,
                "int": 2,
            },
            "str": "abc",
        }
    )
    assert flatten_data == {
        "module.bool": False,
        "module.float": 3.5,
        "module.int": 2,
        "str": "abc",
    }


def test_to_flat_dict_data_empty_key():
    flatten_data = to_flat_dict(
        {
            "module": {},
            "str": "abc",
        }
    )
    assert flatten_data == {"str": "abc"}


def test_to_flat_dict_double_data():
    flatten_data = to_flat_dict(
        {
            "str": "def",
            "module": {
                "component": {
                    "float": 3.5,
                    "int": 2,
                },
            },
        }
    )
    assert flatten_data == {
        "module.component.float": 3.5,
        "module.component.int": 2,
        "str": "def",
    }


def test_to_flat_dict_double_data_2():
    flatten_data = to_flat_dict(
        {
            "module": {
                "component_a": {
                    "float": 3.5,
                    "int": 2,
                },
                "component_b": {
                    "param_a": 1,
                    "param_b": 2,
                },
                "str": "abc",
            },
        }
    )
    assert flatten_data == {
        "module.component_a.float": 3.5,
        "module.component_a.int": 2,
        "module.component_b.param_a": 1,
        "module.component_b.param_b": 2,
        "module.str": "abc",
    }


def test_to_flat_dict_list():
    flatten_data = to_flat_dict([2, "abc", True, 3.5])
    assert flatten_data == {
        "0": 2,
        "1": "abc",
        "2": True,
        "3": 3.5,
    }


def test_to_flat_dict_dict_with_list():
    flatten_data = to_flat_dict(
        {
            "module": [2, "abc", True, 3.5],
            "str": "abc",
        }
    )
    assert flatten_data == {
        "module.0": 2,
        "module.1": "abc",
        "module.2": True,
        "module.3": 3.5,
        "str": "abc",
    }


def test_to_flat_dict_with_more_complex_list():
    flatten_data = to_flat_dict(
        {
            "module": [[1, 2, 3], {"bool": True}],
            "str": "abc",
        }
    )
    assert flatten_data == {
        "module.0.0": 1,
        "module.0.1": 2,
        "module.0.2": 3,
        "module.1.bool": True,
        "str": "abc",
    }


def test_to_flat_dict_tuple():
    flatten_data = to_flat_dict(
        {
            "module": (2, "abc", True, 3.5),
            "str": "abc",
        }
    )
    assert flatten_data == {
        "module.0": 2,
        "module.1": "abc",
        "module.2": True,
        "module.3": 3.5,
        "str": "abc",
    }


def test_to_flat_dict_with_complex_tuple():
    flatten_data = to_flat_dict(
        {
            "module": ([1, 2, 3], {"bool": True}),
            "str": "abc",
        }
    )
    assert flatten_data == {
        "module.0.0": 1,
        "module.0.1": 2,
        "module.0.2": 3,
        "module.1.bool": True,
        "str": "abc",
    }


@mark.parametrize("separator", (".", "/", "@", "[SEP]"))
def test_to_flat_dict_separator(separator: str):
    flatten_data = to_flat_dict(
        {
            "str": "def",
            "module": {
                "component": {
                    "float": 3.5,
                    "int": 2,
                },
            },
        },
        separator=separator,
    )
    assert flatten_data == {
        f"module{separator}component{separator}float": 3.5,
        f"module{separator}component{separator}int": 2,
        "str": "def",
    }


def test_to_flat_dict_to_str_tuple():
    flatten_data = to_flat_dict(
        {
            "module": (2, "abc", True, 3.5),
            "str": "abc",
        },
        to_str=tuple,
    )
    assert flatten_data == {
        "module": "(2, 'abc', True, 3.5)",
        "str": "abc",
    }


def test_to_flat_dict_to_str_tuple_and_list():
    flatten_data = to_flat_dict(
        {
            "module1": (2, "abc", True, 3.5),
            "module2": [1, 2, 3],
            "str": "abc",
        },
        to_str=(list, tuple),
    )
    assert flatten_data == {
        "module1": "(2, 'abc', True, 3.5)",
        "module2": "[1, 2, 3]",
        "str": "abc",
    }


def test_to_flat_dict_tensor():
    assert objects_are_equal(
        to_flat_dict({"tensor": torch.ones(2, 3)}), {"tensor": torch.ones(2, 3)}
    )


def test_to_flat_dict_numpy_ndarray():
    assert objects_are_equal(to_flat_dict(np.zeros((2, 3))), {None: np.zeros((2, 3))})


########################################
#     Tests for to_pretty_json_str     #
########################################


def test_to_pretty_json_str_small_dict():
    assert to_pretty_json_str(data={"my_key": "my_value"}) == "{'my_key': 'my_value'}"


def test_to_pretty_json_str_small_list():
    assert to_pretty_json_str(data=["my_key", "my_value"]) == "['my_key', 'my_value']"


def test_to_pretty_json_str_small_dict_max_len_0():
    assert (
        to_pretty_json_str(data={"my_key": "my_value"}, max_len=0) == '{\n  "my_key": "my_value"\n}'
    )


def test_to_pretty_json_str_small_list_max_len_0():
    assert (
        to_pretty_json_str(data=["my_key", "my_value"], max_len=0)
        == '[\n  "my_key",\n  "my_value"\n]'
    )


def test_to_pretty_json_str_from_str():
    assert to_pretty_json_str(data="abc") == "abc"


def test_to_pretty_json_str_from_tensor():
    assert (
        to_pretty_json_str(data=torch.zeros(2, 3))
        == "tensor([[0., 0., 0.],\n        [0., 0., 0.]])"
    )


########################################
#     Tests for to_pretty_yaml_str     #
########################################


def test_to_pretty_yaml_str_small_dict():
    assert to_pretty_yaml_str(data={"my_key": "my_value"}) == "{'my_key': 'my_value'}"


def test_to_pretty_yaml_str_small_list():
    assert to_pretty_yaml_str(data=["my_key", "my_value"]) == "['my_key', 'my_value']"


def test_to_pretty_yaml_str_small_dict_max_len_0():
    assert (
        to_pretty_yaml_str(data={"my_key1": "my_value1", "my_key2": "my_value2"}, max_len=0)
        == "my_key1: my_value1\nmy_key2: my_value2\n"
    )


def test_to_pretty_yaml_str_small_list_max_len_0():
    assert to_pretty_yaml_str(data=["my_key", "my_value"], max_len=0) == "- my_key\n- my_value\n"


def test_to_pretty_yaml_str_from_str():
    assert to_pretty_yaml_str(data="abc") == "abc"


def test_to_pretty_yaml_str_from_tensor():
    assert (
        to_pretty_yaml_str(data=torch.zeros(2, 3))
        == "tensor([[0., 0., 0.],\n        [0., 0., 0.]])"
    )


########################################
#     Tests for to_pretty_dict_str     #
########################################


def test_to_pretty_dict_str_empty():
    assert to_pretty_dict_str(data={}) == ""


def test_to_pretty_dict_str_1_value():
    assert to_pretty_dict_str(data={"my_key": "my_value"}) == "my_key : my_value"


def test_to_pretty_dict_str_sorted_key():
    assert (
        to_pretty_dict_str(data={"short": 123, "long_key": 2}, sorted_keys=True)
        == "long_key : 2\nshort    : 123"
    )


def test_to_pretty_dict_str_unsorted_key():
    assert (
        to_pretty_dict_str(data={"short": 123, "long_key": 2}, sorted_keys=False)
        == "short    : 123\nlong_key : 2"
    )


def test_to_pretty_dict_str_nested_dict():
    assert to_pretty_dict_str(data={"my_key": {"my_key2": 123}}) == "my_key : {'my_key2': 123}"


def test_to_pretty_dict_str_incorrect_indent():
    with raises(ValueError):
        to_pretty_dict_str(data={"my_key": "my_value"}, indent=-1)


def test_to_pretty_dict_str_indent_2():
    assert to_pretty_dict_str(data={"my_key": "my_value"}, indent=2) == "  my_key : my_value"


##########################################
#     Tests for to_torch_mapping_str     #
##########################################


def test_to_torch_mapping_str_empty():
    assert to_torch_mapping_str({}) == ""


def test_to_torch_mapping_str_1_item():
    assert to_torch_mapping_str({"key": "value"}) == "(key) value"


def test_to_torch_mapping_str_2_items():
    assert (
        to_torch_mapping_str({"key1": "value1", "key2": "value2"}) == "(key1) value1\n(key2) value2"
    )


def test_to_torch_mapping_str_sorted_values_true():
    assert (
        to_torch_mapping_str({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "(key1) value1\n(key2) value2"
    )


def test_to_torch_mapping_str_sorted_values_false():
    assert (
        to_torch_mapping_str({"key2": "value2", "key1": "value1"}) == "(key2) value2\n(key1) value1"
    )


def test_to_torch_mapping_str_2_items_multiple_line():
    assert (
        to_torch_mapping_str({"key1": "abc", "key2": "something\nelse"})
        == "(key1) abc\n(key2) something\n  else"
    )


###########################################
#     Tests for to_torch_sequence_str     #
###########################################


def test_to_torch_sequence_str_empty():
    assert to_torch_sequence_str([]) == ""


def test_to_torch_sequence_str_1_item():
    assert to_torch_sequence_str(["abc"]) == "(0) abc"


def test_to_torch_sequence_str_2_items():
    assert to_torch_sequence_str(["abc", 123]) == "(0) abc\n(1) 123"


def test_to_torch_sequence_str_2_items_multiple_line():
    assert to_torch_sequence_str(["abc", "something\nelse"]) == "(0) abc\n(1) something\n  else"


#################################################
#     Tests for to_human_readable_byte_size     #
#################################################


@mark.parametrize("size,output", ((2, "2.00 B"), (2048, "2,048.00 B"), (2097152, "2,097,152.00 B")))
def test_to_human_readable_byte_size_b(size: int, output: str):
    assert to_human_readable_byte_size(size, "B") == output


@mark.parametrize(
    "size,output",
    [(2048, "2.00 KB"), (2097152, "2,048.00 KB"), (2147483648, "2,097,152.00 KB")],
)
def test_to_human_readable_byte_size_kb(size: int, output: str):
    assert to_human_readable_byte_size(size, "KB") == output


@mark.parametrize(
    "size,output",
    [(2048, "0.00 MB"), (2097152, "2.00 MB"), (2147483648, "2,048.00 MB")],
)
def test_to_human_readable_byte_size_mb(size: int, output: str):
    assert to_human_readable_byte_size(size, "MB") == output


@mark.parametrize("size,output", [(2048, "0.00 GB"), (2097152, "0.00 GB"), (2147483648, "2.00 GB")])
def test_to_human_readable_byte_size_gb(size: int, output: str):
    assert to_human_readable_byte_size(size, "GB") == output


@mark.parametrize(
    "size,output",
    [(2, "2.00 B"), (1023, "1,023.00 B"), (2048, "2.00 KB"), (2097152, "2.00 MB")],
)
def test_to_human_readable_byte_size_auto(size: int, output: str):
    assert to_human_readable_byte_size(size) == output


def test_to_human_readable_byte_size_incorrect_unit():
    with raises(ValueError):
        assert to_human_readable_byte_size(1, "")
