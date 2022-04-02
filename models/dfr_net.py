import torch
import torch.nn as nn
from models.network_tools import conv_block_3d, UNetUp, UNetDown, ConvSig, weight_init


class AvFeat(nn.Module):

    def __init__(self, filter_size):
        super().__init__()
        self.conv1_5x5 = conv_block_3d(1, filter_size, kernel_size=(3, 5, 5), stride=(5, 1, 1), padding=(0, 2, 2))
        self.conv1_3x3 = conv_block_3d(1, filter_size, kernel_size=(3, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1))
        self.conv1_1x1 = conv_block_3d(1, filter_size, kernel_size=(3, 1, 1), stride=(5, 1, 1), padding=0)

        self.conv2_5x5 = conv_block_3d(filter_size, int(filter_size), kernel_size=(3, 5, 5), stride=(5, 1, 1),
                                       padding=(0, 2, 2))
        self.conv2_3x3 = conv_block_3d(filter_size, int(filter_size), kernel_size=(3, 3, 3), stride=(5, 1, 1),
                                       padding=(0, 1, 1))
        self.conv2_1x1 = conv_block_3d(filter_size, int(filter_size), kernel_size=(3, 1, 1), stride=(5, 1, 1),
                                       padding=0)

        self.conv3_5x5 = conv_block_3d(int(filter_size), int(filter_size), kernel_size=(3, 5, 5), stride=(2, 1, 1),
                                       padding=(1, 2, 2))
        self.conv3_3x3 = conv_block_3d(int(filter_size), int(filter_size), kernel_size=(3, 3, 3), stride=(2, 1, 1),
                                       padding=(1, 1, 1))
        self.conv3_1x1 = conv_block_3d(int(filter_size), int(filter_size), kernel_size=(3, 1, 1), stride=(2, 1, 1),
                                       padding=(1, 0, 0))

        # Apply weight initialization
        self.apply(weight_init)

    def forward(self, inp):
        x1_1 = self.conv1_5x5(inp)
        x1_2 = self.conv1_3x3(inp)
        x1_3 = self.conv1_1x1(inp)

        fused1 = (x1_1 + x1_2 + x1_3) / 3

        x2_1 = self.conv2_5x5(fused1)
        x2_2 = self.conv2_3x3(fused1)
        x2_3 = self.conv2_1x1(fused1)

        fused2 = (x2_1 + x2_2 + x2_3) / 3

        x3_1 = self.conv3_5x5(fused2)
        x3_2 = self.conv3_3x3(fused2)
        x3_3 = self.conv3_1x1(fused2)

        fused3 = (x3_1 + x3_2 + x3_3) / 3

        # 5D to 4D tensor
        out = fused3.squeeze(dim=2)

        return out


class ConFeat(nn.Module):

    def __init__(self, filter_size):
        super().__init__()

        self.conv1_5x5 = conv_block_3d(1, filter_size, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.conv1_3x3 = conv_block_3d(1, filter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv1_1x1 = conv_block_3d(1, filter_size, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)

        # Apply weight initialization
        self.apply(weight_init)

    def forward(self, inp):

        x1_1 = self.conv1_5x5(inp)
        x1_2 = self.conv1_3x3(inp)
        x1_3 = self.conv1_1x1(inp)

        x = (x1_1 + x1_2 + x1_3) / 3

        # 5D to 4D tensor
        out = x.squeeze(dim=2)

        return out


class DFR(nn.Module):
        """
        Args:
            inp_ch (int): Number of input channels
            kernel_size (int): Size of the convolutional kernels
            skip (bool, default=True): Use skip connections
        """
        def __init__(self, inp_ch, temporal_length, filter_size, kernel_size=3, skip=True):
            super().__init__()

            self.temporal_length = temporal_length
            self.skip = skip

            inp_ch += 1  # temporal median filter
            self.AvFeat = AvFeat(filter_size=filter_size)
            inp_ch = inp_ch + filter_size

            self.ConFeat = ConFeat(filter_size=filter_size)
            inp_ch = inp_ch + filter_size

            self.enc1 = UNetDown(inp_ch, 64, 2, batch_norm=True, maxpool=False, kernel_size=kernel_size)
            self.enc2 = UNetDown(64, 128, 2, batch_norm=True, maxpool=True, kernel_size=kernel_size)
            self.enc3 = UNetDown(128, 256, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
            self.enc4 = UNetDown(256, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
            self.enc5 = UNetDown(512, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)

            self.dec4 = UNetUp(512, skip * 512, 512, 2, batch_norm=True, kernel_size=kernel_size)
            self.dec3 = UNetUp(512, skip * 256, 256, 2, batch_norm=True, kernel_size=kernel_size)
            self.dec2 = UNetUp(256, skip * 128, 128, 2, batch_norm=True, kernel_size=kernel_size)
            self.dec1 = UNetUp(128, skip * 64, 64, 2, batch_norm=True, kernel_size=kernel_size)
            self.out = ConvSig(64)

            # Apply weight initialization
            self.apply(weight_init)

        def forward(self, inp):
            """
            Args:
                inp (tensor) :              Tensor of input Minibatch

            Returns:
                (tensor): Change detection output
                (tensor): Domain output. Will not be returned when self.adversarial="no"
            """
            # Unsqueeze tensor and give temporal frames to temporal network(AvFeat)
            with torch.no_grad():
                # Get temporal frames
                b, c, h, w = inp.shape
                temporal_median_res = torch.tensor(torch.median(inp, dim=1).values.unsqueeze(dim=1),
                                                   dtype=torch.float32)
                temporal_network_first_index = c - self.temporal_length
                temporal_patch = torch.tensor(inp[:, temporal_network_first_index:],
                                              dtype=torch.float).unsqueeze(dim=1)
                curr_patch = torch.tensor(inp[:, :temporal_network_first_index], dtype=torch.float)

                inp = curr_patch

                avfeat = self.AvFeat(temporal_patch)
                confeat = self.ConFeat(
                    temporal_patch[:, :, -1, :, :].unsqueeze(dim=1))  # give current frame to network

                inp = torch.cat((inp, avfeat, confeat, temporal_median_res), dim=1)

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
