import torch
import torch.nn as nn



def get_mini_model(input_channels = 64, channel_drop = False):

    device = 'cuda'

    class LambdaLayer(nn.Module):
        def __init__(self, lambd):
            super(LambdaLayer, self).__init__()
            self.lambd = lambd

        def forward(self, x):
            return self.lambd(x)

    class ChannelDropout(nn.Module):
        def __init__(self, channels, keep=0.95):
            super().__init__()
            self.channel_mask = torch.nn.Parameter((torch.rand(size=[input_channels]).unsqueeze(-1) < keep) * 1.0,
                                                   requires_grad=False)

        def forward(self, x):
            return x * self.channel_mask

    reg_models = [nn.Sequential(
        LambdaLayer(lambda x: x.permute(0, 2, 1)),
        ChannelDropout(channels=input_channels) if channel_drop else nn.Dropout(p=0.1),
        nn.Conv1d(input_channels, 32, 5, padding='same'),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Conv1d(32, 64, 5, padding='same'),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 128, 5, padding='same'),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Conv1d(128, 1, 1, padding='same'),
        LambdaLayer(lambda x: x.permute(0, 2, 1)),
    ).to(device) for i in range(10)]

    reg_optim = [torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=0) for m in reg_models]

    reg_base = [[torch.tensor(p, requires_grad=False) for p in model.parameters()] for model in reg_models]

    [m.train() for m in reg_models]

    loss_fn = torch.nn.L1Loss()

    reg_models = [m.to(device) for m in reg_models]

    return reg_models, loss_fn, reg_optim, reg_base