"""
U-Net model with attention module
"""

import torch
import torch.nn as nn

from models.unet_tools import UNetDown, UNetUp, ConvSig, FCNN
from models.AvFeat import AvFeat


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class AttU_Net(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)

    def __init__(self, inp_ch):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=inp_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.out = ConvSig(64)

        # Apply weight initialization
        self.apply(self.weight_init)

    class R2AttU_Net(nn.Module):
        def __init__(self, img_ch=3, output_ch=1, t=2):
            super.__init__()

            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.Upsample = nn.Upsample(scale_factor=2)

            self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

            self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

            self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

            self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

            self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

            self.Up5 = up_conv(ch_in=1024, ch_out=512)
            self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
            self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

            self.Up4 = up_conv(ch_in=512, ch_out=256)
            self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
            self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

            self.Up3 = up_conv(ch_in=256, ch_out=128)
            self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
            self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

            self.Up2 = up_conv(ch_in=128, ch_out=64)
            self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
            self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

            self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        def forward(self, x):
            # encoding path
            x1 = self.RRCNN1(x)

            x2 = self.Maxpool(x1)
            x2 = self.RRCNN2(x2)

            x3 = self.Maxpool(x2)
            x3 = self.RRCNN3(x3)

            x4 = self.Maxpool(x3)
            x4 = self.RRCNN4(x4)

            x5 = self.Maxpool(x4)
            x5 = self.RRCNN5(x5)

            # decoding + concat path
            d5 = self.Up5(x5)
            x4 = self.Att5(g=d5, x=x4)
            d5 = torch.cat((x4, d5), dim=1)
            d5 = self.Up_RRCNN5(d5)

            d4 = self.Up4(d5)
            x3 = self.Att4(g=d4, x=x3)
            d4 = torch.cat((x3, d4), dim=1)
            d4 = self.Up_RRCNN4(d4)

            d3 = self.Up3(d4)
            x2 = self.Att3(g=d3, x=x2)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.Up_RRCNN3(d3)

            d2 = self.Up2(d3)
            x1 = self.Att2(g=d2, x=x1)
            d2 = torch.cat((x1, d2), dim=1)
            d2 = self.Up_RRCNN2(d2)

            d1 = self.Conv_1x1(d2)

            return d1

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.out(d2)

        return d1


class AttU_AvFeat_Net(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)
    def __init__(self, inp_ch, temporal_length, filter_size):
        super(AttU_AvFeat_Net, self).__init__()

        self.temporal_length = temporal_length
        self.AvFeat = AvFeat(temporal_length=temporal_length, filter_size=filter_size)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=inp_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.out = ConvSig(64)

    def forward(self, x):

        # Unsqueeze tensor and give temporal frames to temporal network(AvFeat)
        with torch.no_grad():
            # Get temporal frames
            b, c, h, w = x.shape
            temporal_network_first_index = c - self.temporal_length
            temporal_patch = torch.tensor(x[:, temporal_network_first_index:],
                                          dtype=torch.float).unsqueeze(dim=1)
            curr_patch = torch.tensor(x[:, :temporal_network_first_index], dtype=torch.float)

        avfeat = self.AvFeat(temporal_patch)

        # encoding path
        x1 = self.Conv1(torch.cat((curr_patch, avfeat), dim=1))

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.out(d2)

        return d1

