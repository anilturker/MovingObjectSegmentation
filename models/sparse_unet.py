import torch
import torch.nn as nn

from models.temporal_networks import AvFeat, TDR, ConFeat


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
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)


    def __init__(self, inp_ch, temporal_network, temporal_length, filter_size):
        super().__init__()

        self.temporal_network = temporal_network
        self.temporal_length = temporal_length

        if 'avfeat' in self.temporal_network:
            self.AvFeat = AvFeat(filter_size=filter_size)
            inp_ch = inp_ch + filter_size

        if 'confeat' in self.temporal_network:
            self.ConFeat = ConFeat(filter_size=filter_size)
            inp_ch = inp_ch + filter_size

        if 'tdr' in self.temporal_network:
            self.TDR = TDR(inp_ch=temporal_length)
            inp_ch = inp_ch + filter_size

        self.enc1 = conv_block(inp_ch, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.dec1 = up_conv(128, 64)
        self.dec2 = up_conv(64, 32)
        self.dec3 = up_conv(32, 16)

        self.out = ConvSig(16)

        # Apply weight initialization
        self.apply(self.weight_init)

    def forward(self, inp):

        # Unsqueeze tensor and give temporal frames to temporal network(AvFeat)
        if self.temporal_network != 'no':
            with torch.no_grad():
                # Get temporal frames
                b, c, h, w = inp.shape
                temporal_network_first_index = c - self.temporal_length
                temporal_patch = torch.tensor(inp[:, temporal_network_first_index:],
                                              dtype=torch.float).unsqueeze(dim=1)
                curr_patch = torch.tensor(inp[:, :temporal_network_first_index], dtype=torch.float)

            if self.temporal_network in "avfeat":
                temporal_network_res = self.AvFeat(temporal_patch)
            elif self.temporal_network == 'tdr':
                temporal_network_res = self.TDR(temporal_patch.squeeze(dim=1))
            elif self.temporal_network == "avfeat_confeat":
                avfeat = self.AvFeat(temporal_patch)
                confeat = self.ConFeat(temporal_patch[:, :, -1, :, :].unsqueeze(dim=1)) # give current frame to network
                temporal_network_res = torch.cat((avfeat, confeat), dim=1)
            elif self.temporal_network == "avfeat_confeat_tdr":
                avfeat = self.AvFeat(temporal_patch)
                confeat = self.ConFeat(temporal_patch[:, :, -1, :, :].unsqueeze(dim=1)) # give current frame to network
                tdr = self.TDR(temporal_patch.squeeze(dim=1))
                temporal_network_res = torch.cat((avfeat, confeat), dim=1)
                temporal_network_res = torch.cat((temporal_network_res, tdr), dim=1)
            elif self.temporal_network == 'avfeat_tdr':
                avfeat = self.AvFeat(temporal_patch)
                tdr = self.TDR(temporal_patch.squeeze(dim=1))
                temporal_network_res = torch.cat((avfeat, tdr), dim=1)
            else:
                raise ValueError(f"temporal network = {self.temporal_network} is not defined")

            inp = torch.cat((curr_patch, temporal_network_res), dim=1)

        # encoding path
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d1 = self.dec1(e3)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)

        res = self.out(d3)

        return res



