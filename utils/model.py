import torch.nn as nn
import torch


class TSMLPtime(nn.Module):

    def __init__(self, width: int):
        super(TSMLPtime, self).__init__()
        self.lin = nn.Linear(in_features=width, out_features=width)
        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class TSMLPfeat(nn.Module):

    def __init__(self, width: int):
        super(TSMLPfeat, self).__init__()
        self.lin_1 = nn.Linear(in_features=width, out_features=width)
        self.lin_2 = nn.Linear(in_features=width, out_features=width)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_1(x)
        x = self.act(x)
        x = self.dropout_1(x)
        x = self.lin_2(x)
        x = self.dropout_2(x)
        return x


class TSBatchNorm2d(nn.Module):

    def __init__(self):
        super(TSBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)

        # Reshape input_data to (batch_size, 1, timepoints, features)
        x = x.unsqueeze(1)

        # Forward pass
        output = self.bn(x)

        # Reshape the output back to (batch_size, timepoints, features)
        output = output.squeeze(1)
        return output


class TSTimeMixingResBlock(nn.Module):

    def __init__(self, width_time: int):
        super(TSTimeMixingResBlock, self).__init__()
        self.mlp = TSMLPtime(width=width_time)
        self.norm = TSBatchNorm2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.norm(x)
        
        # Now rotate such that shape is (batch_size, features, time)
        y = torch.transpose(y, 1, 2)
        
        # Apply MLP to time dimension
        y = self.mlp(y)
        
        # Rotate back such that shape is (batch_size, time, features)
        y = torch.transpose(y, 1, 2)
        
        # Add residual connection
        return x + y


class TSFeatMixingResBlock(nn.Module):

    def __init__(self, width_feats: int):
        super(TSFeatMixingResBlock, self).__init__()
        self.mlp = TSMLPfeat(width=width_feats)
        self.norm = TSBatchNorm2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.norm(x)
                
        # Apply MLP to feat dimension
        y = self.mlp(y)
                
        # Add residual connection
        return x + y


class TSMixingLayer(nn.Module):

    def __init__(self, input_length: int, no_feats: int):
        super(TSMixingLayer, self).__init__()
        self.time_mixing = TSTimeMixingResBlock(width_time=input_length)
        self.feat_mixing = TSFeatMixingResBlock(width_feats=no_feats)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.time_mixing(x)
        y = self.feat_mixing(y)
        return y


class TSTemporalProjection(nn.Module):

    def __init__(self, input_length: int, forecast_length: int):
        super(TSTemporalProjection, self).__init__()
        self.lin = nn.Linear(in_features=input_length, out_features=forecast_length)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        # Now rotate such that shape is (batch_size, features, time=input_length)
        y = torch.transpose(x, 1, 2)

        # Apply linear projection -> shape is (batch_size, features, time=forecast_length)
        y = self.lin(y)

        # Rotate back such that shape is (batch_size, time=forecast_length, features)
        y = torch.transpose(y, 1, 2)
        return y


class TSMixerModel(nn.Module):

    def __init__(self, input_length: int, forecast_length: int, no_feats: int, no_mixer_layers: int):
        super(TSMixerModel, self).__init__()
        self.temp_proj = TSTemporalProjection(input_length=input_length, forecast_length=forecast_length)
        self.mixer_layers = []
        for _ in range(no_mixer_layers):
            self.mixer_layers.append(TSMixingLayer(input_length=input_length, no_feats=no_feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)

        # Apply temporal projection -> shape is (batch_size, time=forecast_length, features)
        x = self.temp_proj(x)

        return x