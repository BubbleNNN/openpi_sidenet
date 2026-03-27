'''
This File contains the implementation of the CNN Encoder, which might be helpful in encoding the temporal modality.
'''
from torch import nn
from torch.nn import functional as F

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride=1, padding=1, norm = None, gn_groups = None):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, kernel_size, stride, padding)
        self.norm = norm
        if self.norm is not None:
            if norm == 'gn':
                if gn_groups is None:
                    raise ValueError("gn_groups must be specified for group normalization")
                assert isinstance(gn_groups, int)
                assert mid_channels % gn_groups == 0, "mid_channels must be divisible by gn_groups"
                assert out_channels % gn_groups == 0, "out_channels must be divisible by gn_groups"
                self.norm1 = nn.GroupNorm(gn_groups, mid_channels)
                self.norm2 = nn.GroupNorm(gn_groups, out_channels)
            elif norm == 'bn':
                self.norm1 = nn.BatchNorm1d(mid_channels)
                self.norm2 = nn.BatchNorm1d(out_channels)
            elif norm == 'ln':
                self.norm1 = nn.LayerNorm(mid_channels)
                self.norm2 = nn.LayerNorm(out_channels)
            else:
                raise ValueError(f"Unknown normalization type: {norm}")
        '''
        Here, the in_channels should match the sensor_dim of the input data
        The kernel will slide over the time dimension, 
        dot product each sensor dimension with the kernel(different between dimensions), 
        and add them together to get a scalar, which corresponds to one of the elements in the output_channels
        '''
        

    def forward(self, x):
        # x shape: (batch_size, time_steps, sensor_dim)
        x = x.transpose(1, 2)  # (batch_size, sensor_dim, time_steps)
        x = self.conv1(x)  # (batch_size, mid_channels, time_steps)
        if self.norm is not None:
            if self.norm == 'ln':
                # For LayerNorm, we need to transpose back to (batch_size, time_steps, mid_channels) before applying it
                x = x.transpose(1, 2)
                x = self.norm1(x)
                x = x.transpose(1, 2)
            else:
                x = self.norm1(x)
        x = F.gelu(x)
        x = self.conv2(x)  # (batch_size, out_channels, time_steps)
        if self.norm is not None:
            if self.norm == 'ln':
                x = x.transpose(1, 2)
                x = self.norm2(x)
                x = x.transpose(1, 2)
            else:
                x = self.norm2(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)  # (batch_size, time_steps, out_channels)
        return x