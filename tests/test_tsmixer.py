import sys
sys.path.append("..")

from utils import TSMixer

import pytest
import torch
import os
import shutil

TEST_CSV_NO_FEATS = 7


@pytest.fixture
def conf():
    output_dir = "TMP_output_dir"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    yield TSMixer.Conf(
        input_length=20,
        prediction_length=5,
        no_features=TEST_CSV_NO_FEATS,
        no_mixer_layers=2,
        output_dir=output_dir,
        data_src=TSMixer.Conf.DataSrc.CSV_FILE,
        data_src_csv="test_csv.csv",
        batch_size=4,
        num_epochs=10,
        learning_rate=0.001,
        optimizer="Adam"
        )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    

@pytest.fixture
def tsmixer(conf: TSMixer.Conf):
    return TSMixer(conf=conf)


class TestTsMixer:

    def test_load_data(self, tsmixer: TSMixer):
        loader_train, loader_val = tsmixer.load_data_train_val()
        batch = next(iter(loader_train))
        assert batch.shape == (tsmixer.conf.batch_size, tsmixer.conf.input_length, TEST_CSV_NO_FEATS)

        batch = next(iter(loader_val))
        assert batch.shape == (tsmixer.conf.batch_size, tsmixer.conf.input_length, TEST_CSV_NO_FEATS)