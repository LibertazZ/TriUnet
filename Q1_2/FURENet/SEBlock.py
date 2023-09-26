import torch
from torch import nn, Tensor
import torch.nn.functional as F

__all__ = ["SEBlock"]


class SEBlock(nn.Module):
    def __init__(self, channels: int):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 1),
                      stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x: Tensor):
        reassigned_weights = self.fc(x)
        x = x * reassigned_weights
        return x
    

class FiLM_Layer(nn.Module):
    def __init__(self, channels, in_channels=3, alpha=1, activation=F.leaky_relu):
        '''
        input size: (N, in_channels). output size: (N, channels)
        
        Args:
            channels: int.
            alpha: scalar. Expand ratio for FiLM hidden layer.
        '''
        super(FiLM_Layer, self).__init__()
        self.channels = channels
        self.activation = activation
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, alpha*channels*2, bias=True), 
            nn.LeakyReLU(inplace=True),
            nn.Linear(alpha*channels*2, channels*2, bias=True), 
        )
        
    def forward(self, _input, _lambda):
        N, C, H, W = _input.size()
        # print(_lambda)
        out = self.MLP(_lambda)
        self.mu, self.sigma = torch.split(out, [self.channels, self.channels], dim=-1)
        if self.activation is not None:
            self.mu, self.sigma = self.activation(self.mu), self.activation(self.sigma)
        _output = _input * self.mu.view(N, C, 1, 1).expand_as(_input) + self.sigma.view(N, C, 1, 1).expand_as(_input)
        return _output



# if __name__ == '__main__':
#     a = torch.ones(3, 512, 4, 4)
#     net = SEBlock(channels=512)
#     r = net(a)
#     print(r.shape)
