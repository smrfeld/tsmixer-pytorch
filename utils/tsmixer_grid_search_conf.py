from .tsmixer_conf import TSMixerConf

from dataclasses import dataclass, field
from mashumaro import DataClassDictMixin
from typing import Optional, Tuple, Dict, List, Iterator
from loguru import logger
import os


@dataclass
class TSMixerGridSearch(DataClassDictMixin):
    """Configuration for grid search
    """    

    @dataclass
    class ParamRange(DataClassDictMixin):
        
        learning_rates: List[float]
        "Learning rates"

        no_mixer_layers: List[int]
        "Number of mixer layers"

        dropouts: List[float]
        "Dropout"

        input_lengths: List[int]
        "Number of time steps to use as input"

        prediction_lengths: List[int]
        "Number of time steps to predict"

        feat_mixing_hidden_channels: List[Optional[int]] = field(default_factory=lambda: [None])
        "Number of hidden channels in the feature mixing MLP. If None, uses same as input features."

        batch_sizes: List[int] = field(default_factory=lambda: [64])
        "Batch size"

        num_epochs: List[int] = field(default_factory=lambda: [100])
        "Number of epochs to train for"

        optimizers: List[str] = field(default_factory=lambda: ["Adam"])
        "Optimizer to use"

        @property
        def options_str(self) -> str:
            s = []
            s.append(("lr",str(self.learning_rates)))
            s.append(("nmix",str(self.no_mixer_layers)))
            s.append(("drop",str(self.dropouts)))
            s.append(("in",str(self.input_lengths)))
            s.append(("pred",str(self.prediction_lengths)))
            s.append(("hidden",str(self.feat_mixing_hidden_channels)))
            s.append(("batch",str(self.batch_sizes)))
            s.append(("epochs",str(self.num_epochs)))
            s.append(("opt",str(self.optimizers)))

            # Sort by key
            s = sorted(s, key=lambda x: x[0])

            return "_".join([f"{k}{v}" for k,v in s])

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
        """Iterate over all configurations

        Yields:
            Iterator[TSMixerConf]: Configuration for a single run
        """        
        for idx,param_range in enumerate(self.param_ranges):
            logger.info("===========================================")
            logger.info(f"Grid search iteration {idx+1}/{len(self.param_ranges)}")
            logger.info("===========================================")

            for learning_rate in param_range.learning_rates:
                for no_mixer_layers in param_range.no_mixer_layers:
                    for dropout in param_range.dropouts:
                        for feat_mixing_hidden_channels in param_range.feat_mixing_hidden_channels:
                            for input_length in param_range.input_lengths:
                                for prediction_length in param_range.prediction_lengths:
                                    for batch_size in param_range.batch_sizes:
                                        for num_epochs in param_range.num_epochs:
                                            for optimizer in param_range.optimizers:
                                                # Output subdir
                                                output_dir = os.path.join(self.output_dir, param_range.options_str)
                                                conf = TSMixerConf(
                                                    input_length=input_length,
                                                    prediction_length=prediction_length,
                                                    no_features=self.no_features,
                                                    no_mixer_layers=no_mixer_layers,
                                                    output_dir=output_dir,
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
                                                logger.info(f"Output sub-dir: {output_dir}")
                                                yield conf
