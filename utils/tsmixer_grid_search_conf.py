from .tsmixer_conf import TSMixerConf

from dataclasses import dataclass, field
from mashumaro import DataClassDictMixin
from typing import Optional, Tuple, Dict, List, Iterator
from loguru import logger


@dataclass
class TSMixerGridSearch(DataClassDictMixin):

    @dataclass
    class ParamRange(DataClassDictMixin):
        
        learning_rate: List[float]
        "Learning rates"

        no_mixer_layers: List[int]
        "Number of mixer layers"

        dropout: List[float]
        "Dropout"

        input_length: List[int]
        "Number of time steps to use as input"

        prediction_length: List[int]
        "Number of time steps to predict"

        feat_mixing_hidden_channels: List[Optional[int]] = field(default_factory=lambda: [None])
        "Number of hidden channels in the feature mixing MLP. If None, uses same as input features."

        batch_size: List[int] = field(default_factory=lambda: [64])
        "Batch size"

        num_epochs: List[int] = field(default_factory=lambda: [100])
        "Number of epochs to train for"

        optimizer: List[str] = field(default_factory=lambda: ["Adam"])
        "Optimizer to use"

    param_ranges: List[ParamRange]
    "Any number of parameter ranges to try"

    output_dir: str
    "Output directory"

    no_features: int
    "Number of features in the dataset"

    data_src: TSMixerConf.DataSrc = TSMixerConf.DataSrc.CSV_FILE
    "Where to load the dataset from"

    data_src_csv: Optional[str] = None
    "Path to the CSV file to load the dataset from. Only used if data_src is CSV_FILE"

    def iterate(self) -> Iterator[TSMixerConf]:
        for idx,param_range in enumerate(self.param_ranges):
            logger.info("===========================================")
            logger.info(f"Grid search iteration {idx+1}/{len(self.param_ranges)}")
            logger.info("===========================================")

            for learning_rate in param_range.learning_rate:
                for no_mixer_layers in param_range.no_mixer_layers:
                    for dropout in param_range.dropout:
                        for feat_mixing_hidden_channels in param_range.feat_mixing_hidden_channels:
                            for input_length in param_range.input_length:
                                for prediction_length in param_range.prediction_length:
                                    for batch_size in param_range.batch_size:
                                        for num_epochs in param_range.num_epochs:
                                            for optimizer in param_range.optimizer:
                                                conf = TSMixerConf(
                                                    input_length=input_length,
                                                    prediction_length=prediction_length,
                                                    no_features=self.no_features,
                                                    no_mixer_layers=no_mixer_layers,
                                                    output_dir=self.output_dir,
                                                    data_src=self.data_src,
                                                    data_src_csv=self.data_src_csv,
                                                    batch_size=batch_size,
                                                    num_epochs=num_epochs,
                                                    learning_rate=learning_rate,
                                                    optimizer=optimizer,
                                                    dropout=dropout,
                                                    feat_mixing_hidden_channels=feat_mixing_hidden_channels
                                                    )
                                                logger.info(f"TSMixer config: {conf}")
                                                yield conf
