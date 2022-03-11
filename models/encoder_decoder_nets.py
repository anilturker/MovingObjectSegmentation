import torch
import torch.nn as nn

from models.temporal_networks import AvFeat


class conv_block(nn.Module):
    def __init__(self,ch_in, ch_out, kernel_size=3):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=2, padding=0, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=1), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.up(x)
        return x


class ConvSig(nn.Module):
    """ Conv layer + Sigmoid

    Args:
        in_ch (int): Number of input channels
    """

    def __init__(self, in_ch):
        super().__init__()
        self.out = nn.Sequential()
        self.out.add_module("conv2d", nn.Conv2d(in_ch, 1, 1))
        self.out.add_module("sigmoid", nn.Sigmoid())

    def forward(self, inp):
        return self.out(inp)


class FgNet(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)

    def __init__(self, inp_ch, temporal_length, filter_size):
        super(FgNet, self).__init__()

        self.temporal_length = temporal_length
        self.AvFeat = AvFeat(filter_size=filter_size)

        self.enc1 = conv_block(inp_ch, 8)
        self.enc2 = conv_block(8, 4)
        self.dec1 = up_conv(4, 8)
        self.dec2 = up_conv(8, 8)

        self.out = ConvSig(8)

        # Apply weight initialization
        self.apply(self.weight_init)

    def forward(self, inp):

        # Unsqueeze tensor and give temporal frames to temporal network(AvFeat)
        with torch.no_grad():
            # Get temporal frames
            b, c, h, w = inp.shape
            temporal_network_first_index = c - self.temporal_length
            temporal_patch = torch.tensor(inp[:, temporal_network_first_index:],
                                          dtype=torch.float).unsqueeze(dim=1)
            curr_patch = torch.tensor(inp[:, :temporal_network_first_index], dtype=torch.float)

        avfeat = self.AvFeat(temporal_patch)

        e1 = self.enc1(torch.cat((curr_patch, avfeat), dim=1))
        e2 = self.enc2(e1)
        d1 = self.dec1(e2)
        d2 = self.dec2(d1)

        res = self.out(d2)

        return res



