"""
16-layer U-Net model
"""
import torch
import torch.nn as nn
from models.unet_tools import UNetDown, UNetUp, ConvSig
from models.temporal_networks import AvFeat, TDR

class unet_vgg16(nn.Module):
    """
    Args:
        inp_ch (int): Number of input channels
        kernel_size (int): Size of the convolutional kernels
        skip (bool, default=True): Use skip connections
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)

    def __init__(self, inp_ch, temporal_network, temporal_length, filter_size, kernel_size=3, skip=True):
        super().__init__()

        self.temporal_network = temporal_network
        self.temporal_length = temporal_length
        self.skip = skip

        if 'avfeat' in self.temporal_network:
            self.AvFeat = AvFeat(filter_size=filter_size)
            inp_ch = inp_ch + filter_size

        if 'tdr' in self.temporal_network:
            self.TDR = TDR(inp_ch=temporal_length)
            inp_ch = inp_ch + 1

        self.enc1 = UNetDown(inp_ch, 64, 2, batch_norm=True, maxpool=False, kernel_size=kernel_size)
        self.enc2 = UNetDown(64, 128, 2, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc3 = UNetDown(128, 256, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc4 = UNetDown(256, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc5 = UNetDown(512, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)

        self.dec4 = UNetUp(512, skip*512, 512, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(512, skip*256, 256, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(256, skip*128, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(128, skip*64, 64, 2, batch_norm=True, kernel_size=kernel_size)
        self.out = ConvSig(64)

        self.frozenLayers = [self.enc1, self.enc2, self.enc3]

        # Apply weight initialization
        self.apply(self.weight_init)

    def forward(self, inp):
        """
        Args:
            inp (tensor) :              Tensor of input Minibatch

        Returns:
            (tensor): Change detection output
            (tensor): Domain output. Will not be returned when self.adversarial="no"
        """
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
            elif self.temporal_network == 'avfeat_tdr':
                avfeat = self.AvFeat(temporal_patch)
                tdr = self.TDR(temporal_patch.squeeze(dim=1))
                temporal_network_res = torch.cat((avfeat, tdr), dim=1)
            else:
                raise ValueError(f"temporal network = {self.temporal_network} is not defined")

            inp = torch.cat((curr_patch, temporal_network_res), dim=1)

        # encoding path
        d1 = self.enc1(inp)
        d2 = self.enc2(d1)
        d3 = self.enc3(d2)
        d4 = self.enc4(d3)
        d5 = self.enc5(d4)
        if self.skip:
            u4 = self.dec4(d5, d4)
            u3 = self.dec3(u4, d3)
            u2 = self.dec2(u3, d2)
            u1 = self.dec1(u2, d1)
        else:
            u4 = self.dec4(d5)
            u3 = self.dec3(u4)
            u2 = self.dec2(u3)
            u1 = self.dec1(u2)
            
        cd_out = self.out(u1)
        return cd_out
