from unittest.mock import Mock, patch

from gravitorch.engines import BaseEngine
from gravitorch.utils.parameter_initializers import BaseDefaultParameterInitializer

#####################################################
#     Tests for BaseDefaultParameterInitializer     #
#####################################################


class FakeParameterInitializer(BaseDefaultParameterInitializer):
    def _initialize(self, engine: BaseEngine) -> None:
        pass


def test_base_default_model_parameter_initializer_show_stats_true() -> None:
    engine = Mock()
    with patch("gravitorch.utils.parameter_initializers.base.show_parameter_stats") as mock_show:
        parameter_initializer = FakeParameterInitializer()
        parameter_initializer.initialize(engine)
        mock_show.assert_called_once_with(engine.model)


def test_base_default_model_parameter_initializer_show_stats_false() -> None:
    engine = Mock()
    with patch("gravitorch.utils.parameter_initializers.base.show_parameter_stats") as mock_show:
        parameter_initializer = FakeParameterInitializer(show_stats=False)
        parameter_initializer.initialize(engine)
        mock_show.assert_not_called()
