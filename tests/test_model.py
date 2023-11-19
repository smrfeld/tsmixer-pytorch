import sys
sys.path.append("..")

from utils import TSMixerModelExclRIN, TSBatchNorm2d, TSFeatMixingResBlock, TSMixingLayer, TSTemporalProjection, TSTimeMixingResBlock, TSMixerModel

import pytest
import torch

class TestModel:

    def _time_series(self, batch_size: int, input_length: int, no_feats: int) -> torch.Tensor:
        return torch.randn(batch_size, input_length, no_feats)

    def test_batchnorm2d(self):
        bn = TSBatchNorm2d()
        data = self._time_series(batch_size=32, input_length=100, no_feats=5)
        data_out = bn(data)
        assert data_out.shape == data.shape

    
    def test_tstemporalprojection(self):
        tp = TSTemporalProjection(input_length=100, forecast_length=30)
        data = self._time_series(batch_size=32, input_length=100, no_feats=5)
        data_out = tp(data)
        assert data_out.shape == (32, 30, 5)


    def test_tsmixinglayer(self):
        ml = TSMixingLayer(input_length=100, no_feats=5, dropout=0.5, feat_mixing_hidden_channels=10)
        data = self._time_series(batch_size=32, input_length=100, no_feats=5)
        data_out = ml(data)
        assert data_out.shape == data.shape


    def test_tsfeatmixingresblock(self):
        fmrb = TSFeatMixingResBlock(width_feats=5, dropout=0.5, width_feats_hidden=10)
        data = self._time_series(batch_size=32, input_length=100, no_feats=5)
        data_out = fmrb(data)
        assert data_out.shape == data.shape


    def test_tstimemixingresblock(self):
        tmrb = TSTimeMixingResBlock(width_time=100, dropout=0.5)
        data = self._time_series(batch_size=32, input_length=100, no_feats=5)
        data_out = tmrb(data)
        assert data_out.shape == data.shape


    def test_tsmixer_excl_rin(self):

        ts = TSMixerModelExclRIN(
            input_length=100,
            forecast_length=10,
            no_feats=5,
            no_mixer_layers=3,
            dropout=0.5,
            feat_mixing_hidden_channels=5
            )
        data = self._time_series(batch_size=32, input_length=100, no_feats=5)
        forecast = ts(data)

        assert forecast.shape == (32, 10, 5)


    def test_tsmixer(self):

        ts = TSMixerModel(
            input_length=100,
            forecast_length=10,
            no_feats=5,
            no_mixer_layers=3,
            dropout=0.5,
            feat_mixing_hidden_channels=5
            )
        data = self._time_series(batch_size=32, input_length=100, no_feats=5)
        forecast = ts(data)

        assert forecast.shape == (32, 10, 5)
