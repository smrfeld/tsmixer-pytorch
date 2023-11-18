import sys
sys.path.append("..")

from utils import TSMixer

import pytest
import torch

class TestTsMixer:

    def _time_series(self, batch_size: int, input_length: int, no_feats: int) -> torch.Tensor:
        return torch.randn(batch_size, input_length, no_feats)

    def test_tsmixer(self):

        ts = TSMixer(
            input_length=100,
            forecast_length=10,
            no_feats=5,
            no_mixer_layers=3
            )
        data = self._time_series(batch_size=32, input_length=100, no_feats=5)
        forecast = ts(data)

        assert forecast.shape == (32, 10, 5)
