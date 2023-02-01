from gravitorch.utils.sysinfo import virtual_memory_human_summary
from tests.testing import psutil_available

##################################################
#     Tests for virtual_memory_human_summary     #
##################################################


def test_virtual_memory_human_summary():
    assert isinstance(virtual_memory_human_summary(), str)


@psutil_available
def test_virtual_memory_human_summary_with_psutil():
    assert virtual_memory_human_summary().startswith("virtual memory")
