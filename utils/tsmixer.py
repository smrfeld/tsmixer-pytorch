import torch.nn as nn
import torch

class MLPtime(nn.Module):

    def __init__(self, width: int):
        self.lin = nn.Linear(in_features=width, out_features=width)
        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class MLPfeat(nn.Module):

    def __init__(self, width: int):
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


class TimeMixingResBlock(nn.Module):

    def __init__(self, width_time: int):
        self.mlp = MLPtime(width=width_time)
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

class FeatMixingResBlock(nn.Module):

    def __init__(self, width_feats: int):
        self.mlp = MLPfeat(width=width_feats)
        self.norm = TSBatchNorm2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.norm(x)
                
        # Apply MLP to feat dimension
        y = self.mlp(y)
                
        # Add residual connection
        return x + y

class TemporalProjection(nn.Module):

    def __init__(self, input_length: int, forecast_length: int):
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