import torch
import torch.nn as nn
from models.convlstm_network import ConvLSTMBlock


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class conv_block_3d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                 dilation=(1,1,1), padding=(0, 0, 0)):
        super(conv_block_3d,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride,
                      dilation=dilation, padding=padding, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in, ch_out, maxpool, kernel_size, stride, padding):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("conv2d", nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                       bias=True))
        if maxpool:
            self.conv.add_module("maxpool2d", nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv.add_module("bn2d", nn.BatchNorm2d(ch_out))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class AvFeat(nn.Module):

    def __init__(self, filter_size):
        super().__init__()
        self.conv1_5x5 = conv_block_3d(1, filter_size, kernel_size=(3, 5, 5), stride=(5, 1, 1), padding=(0, 2, 2))
        self.conv1_3x3 = conv_block_3d(1, filter_size, kernel_size=(3, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1))
        self.conv1_1x1 = conv_block_3d(1, filter_size, kernel_size=(3, 1, 1), stride=(5, 1, 1), padding=0)

        # Temporal depth 10
        self.conv1_avg_1x1 = conv_block_3d(filter_size * 3, filter_size, 1, stride=(1, 1, 1), padding=0)

        self.conv2_5x5 = conv_block_3d(filter_size, int(filter_size), kernel_size=(3, 5, 5), stride=(5, 1, 1), padding=(0, 2, 2))
        self.conv2_3x3 = conv_block_3d(filter_size, int(filter_size), kernel_size=(3, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1))
        self.conv2_1x1 = conv_block_3d(filter_size, int(filter_size), kernel_size=(3, 1, 1), stride=(5, 1, 1), padding=0)

        self.conv2_avg_1x1 = conv_block_3d(int(filter_size * 3), int(filter_size), 1, stride=(1, 1, 1), padding=0)

        self.conv3_5x5 = conv_block_3d(int(filter_size), int(filter_size), kernel_size=(3, 5, 5), stride=(2, 1, 1), padding=(1, 2, 2))
        self.conv3_3x3 = conv_block_3d(int(filter_size), int(filter_size), kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.conv3_1x1 = conv_block_3d(int(filter_size), int(filter_size), kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.conv3_avg_1x1 = conv_block_3d(int(filter_size * 3), int(filter_size), 1, stride=(1, 1, 1), padding=0)

        self.convlstm_block = ConvLSTMBlock(filter_size*3, filter_size, kernel_size=3, padding=1)

        # Apply weight initialization
        self.apply(weight_init)

    def forward(self, inp):
        x1_1 = self.conv1_5x5(inp)
        x1_2 = self.conv1_3x3(inp)
        x1_3 = self.conv1_1x1(inp)

        fused1 = torch.cat((x1_1, x1_2, x1_3), dim=1)
        # fused1 = self.conv1_avg_1x1(fused1)
        fused1 = self.convlstm_block(fused1)

        # fused1 = (x1_1 + x1_2 + x1_3) / 3

        x2_1 = self.conv2_5x5(fused1)
        x2_2 = self.conv2_3x3(fused1)
        x2_3 = self.conv2_1x1(fused1)

        fused2 = torch.cat((x2_1, x2_2, x2_3), dim=1)
        # fused2 = self.conv2_avg_1x1(fused2)
        fused2 = self.convlstm_block(fused2)

        # fused2 = (x2_1 + x2_2 + x2_3) / 3

        x3_1 = self.conv3_5x5(fused2)
        x3_2 = self.conv3_3x3(fused2)
        x3_3 = self.conv3_1x1(fused2)

        fused3 = torch.cat((x3_1, x3_2, x3_3), dim=1)
        fused3 = self.conv3_avg_1x1(fused3)

        # fused3 = (x3_1 + x3_2 + x3_3) / 3

        # 5D to 4D tensor
        with torch.no_grad():
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
        with torch.no_grad():
            out = x.squeeze(dim=2)

        return out

class TDR(nn.Module):
    """
    implements:
        background estimation using temporal depth reduction
        inputs:
            image_fame: history image frame for background computation
        return:
            background estimation model
    """
    def __init__(self, inp_ch):
        super().__init__()
        self.tdr_layer_1_1 = conv_block(inp_ch, 32, maxpool=False, kernel_size=(1, 1), stride=1, padding=0)
        self.tdr_layer_1_2 = conv_block(inp_ch, 32, maxpool=False, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.tdr_layer_1_3 = conv_block(inp_ch, 32, maxpool=False, kernel_size=(5, 5), stride=1, padding=(2, 2))

        self.tdr_layer_1 = conv_block(32 * 3, 32, maxpool=False, kernel_size=(1, 1), stride=1, padding=0)

        self.tdr_layer_2_1 = conv_block(32, 16, maxpool=False, kernel_size=(1, 1), stride=1, padding=0)
        self.tdr_layer_2_2 = conv_block(32, 16, maxpool=False, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.tdr_layer_2_3 = conv_block(32, 16, maxpool=False, kernel_size=(5, 5), stride=1, padding=(2, 2))

        self.tdr_layer_2 = conv_block(16 * 3, 16, maxpool=False, kernel_size=(1, 1), stride=1, padding=0)

        self.tdr_layer_3_1 = conv_block(16, 8, maxpool=False, kernel_size=(1, 1), stride=1, padding=0)
        self.tdr_layer_3_2 = conv_block(16, 8, maxpool=False, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.tdr_layer_3_3 = conv_block(16, 8, maxpool=False, kernel_size=(5, 5), stride=1, padding=(2, 2))

        self.tdr_layer_3 = conv_block(8 * 3, 8, maxpool=False, kernel_size=(1, 1), stride=1, padding=0)

        # self.tdr_layer_4 = conv_block(8, 1, maxpool=False, kernel_size=(3, 3), stride=1, padding=1)

        # Apply weight initialization
        self.apply(self.weight_init)

    def forward(self, inp):

        x1_1 = self.tdr_layer_1_1(inp)
        x1_2 = self.tdr_layer_1_2(inp)
        x1_3 = self.tdr_layer_1_3(inp)

        fused1 = torch.cat((x1_1, x1_2, x1_3), dim=1)
        fused1 = self.tdr_layer_1(fused1)

        x2_1 = self.tdr_layer_2_1(fused1)
        x2_2 = self.tdr_layer_2_2(fused1)
        x2_3 = self.tdr_layer_2_3(fused1)

        fused2 = torch.cat((x2_1, x2_2, x2_3), dim=1)
        fused2 = self.tdr_layer_2(fused2)

        x3_1 = self.tdr_layer_3_1(fused2)
        x3_2 = self.tdr_layer_3_2(fused2)
        x3_3 = self.tdr_layer_3_1(fused2)

        fused3 = torch.cat((x3_1, x3_2, x3_3), dim=1)
        out = self.tdr_layer_3(fused3)

        return out


class M_FPM(nn.Module):
    """

    """
    def __init__(self, filter_size=8):
        super().__init__()

        in_ch = 1
        in_ch_for_conv = in_ch
        self.fpm_block_1 = nn.Sequential()
        #self.fpm_block_1.add_module("zero pad", nn.ZeroPad2d((0, 1, 0, 1)))
        #self.fpm_block_1.add_module("pool", nn.MaxPool2d(kernel_size=(2,2), stride=(1, 1)))
        self.fpm_block_1.add_module("conv2d", nn.Conv2d(in_ch_for_conv, filter_size,
                                                        kernel_size=(1, 1)))

        self.fpm_block_2 = nn.Sequential()
        self.fpm_block_2.add_module("conv2d", nn.Conv2d(in_ch_for_conv, filter_size,
                                                        kernel_size=(3, 3), padding=(1, 1)))

        in_ch_for_conv += filter_size
        dilation = (4, 4)
        self.fpm_block_3 = nn.Sequential()
        self.fpm_block_3.add_module("act", nn.ReLU())
        self.fpm_block_3.add_module("conv2d", nn.Conv2d(in_ch_for_conv, filter_size,
                                                        kernel_size=(3, 3), dilation=dilation,
                                                        padding=dilation))
        dilation = (8, 8)
        self.fpm_block_4 = nn.Sequential()
        self.fpm_block_4.add_module("act", nn.ReLU())
        self.fpm_block_4.add_module("conv2d", nn.Conv2d(in_ch_for_conv, filter_size,
                                                        kernel_size=(3, 3), dilation=dilation,
                                                        padding=dilation))
        dilation = (16, 16)
        self.fpm_block_5 = nn.Sequential()
        self.fpm_block_5.add_module("act", nn.ReLU())
        self.fpm_block_5.add_module("conv2d", nn.Conv2d(in_ch_for_conv, filter_size,
                                                        kernel_size=(3, 3), dilation=dilation,
                                                        padding=dilation))

        in_ch_for_conv = filter_size * 5
        self.fpm_out_block = nn.Sequential()
        self.fpm_out_block.add_module("conv2d", nn.Conv2d(in_ch_for_conv, filter_size, kernel_size=(1, 1)))
        self.fpm_out_block.add_module("inst_norm", nn.BatchNorm2d(filter_size))
        self.fpm_out_block.add_module("act", nn.ReLU())
        self.fpm_out_block.add_module("dropout", nn.Dropout2d(0.25))

        # Apply weight initialization
        self.apply(weight_init)

    def forward(self, inp):
        res_1 = self.fpm_block_1(inp)
        res_2 = self.fpm_block_2(inp)

        merged = torch.cat([inp, res_2], dim=1)
        res_3 = self.fpm_block_3(merged)

        merged = torch.cat([inp, res_3], dim=1)
        res_4 = self.fpm_block_4(merged)

        merged = torch.cat([inp, res_4], dim=1)
        res_5 = self.fpm_block_5(merged)

        merged = torch.cat([res_1, res_2, res_3, res_4, res_5], dim=1)
        fpm_out = self.fpm_out_block(merged)

        return fpm_out


class AvShortFeat(nn.Module):

    def __init__(self, filter_size):
        super().__init__()

        self.conv3D11 = conv_block_3d(1, 64, kernel_size=(6, 3, 3), stride=(1, 1, 1),
                                      dilation=(1, 1, 1), padding=(0, 1, 1))
        self.conv3D12 = conv_block_3d(1, 64, kernel_size=(6, 3, 3), stride=(1, 1, 1),
                                      dilation=(1, 1, 1), padding=(0, 1, 1))
        self.conv3D13 = conv_block_3d(1, 64, kernel_size=(6, 3, 3), stride=(1, 1, 1),
                                      dilation=(1, 1, 1), padding=(0, 1, 1))

        self.conv3D14 = conv_block_3d(64, 128, kernel_size=(2, 3, 3), stride=(1, 1, 1),
                                      dilation=(1, 2, 2), padding=(0, 2, 2))
        self.conv3D15 = conv_block_3d(64, 128, kernel_size=(2, 3, 3), stride=(1, 1, 1),
                                      dilation=(1, 2, 2), padding=(0, 2, 2))

        self.conv3D16 = conv_block_3d(128, 256, kernel_size=(2, 3, 3), stride=(1, 1, 1),
                                      dilation=(1, 4, 4), padding=(0, 4, 4))

        self.conv3D21 = conv_block_3d(1, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                      dilation=(4, 1, 1), padding=(0, 1, 1))
        self.conv3D22 = conv_block_3d(64, 128, kernel_size=(2, 3, 3), stride=(1, 1, 1),
                                      dilation=(2, 2, 2), padding=(0, 2, 2))
        self.conv3D23 = conv_block_3d(128, 256, kernel_size=(2, 3, 3), stride=(1, 1, 1),
                                      dilation=(1, 4, 4), padding=(0, 4, 4))

        self.conv3D31 = conv_block_3d(256, 256, kernel_size=(2, 3, 3), stride=(1, 1, 1),
                                      dilation=(1, 1, 1), padding=(0, 1, 1))

        # Temporal depth 10
        self.conv2d_1 = conv_block(256, 128, maxpool=False, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_2 = conv_block(128, 64, maxpool=False, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_3 = conv_block(64, 32, maxpool=False, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_4 = conv_block(32, filter_size, maxpool=False, kernel_size=(3, 3), stride=1, padding=1)

        # Apply weight initialization
        self.apply(weight_init)

    def forward(self, inp):
        x1_1 = self.conv3D11(inp[:, :, :6])
        x1_2 = self.conv3D12(inp[:, :, 3:9])
        x1_3 = self.conv3D13(inp[:, :, 6:])

        fused_1_1 = torch.cat((x1_1, x1_2), dim=2)
        fused_1_2 = torch.cat((x1_2, x1_3), dim=2)

        x1_4 = self.conv3D14(fused_1_1)
        x1_5 = self.conv3D14(fused_1_2)

        fused_2 = torch.cat((x1_4, x1_5), dim=2)
        x1_6 = self.conv3D16(fused_2)

        x2_1 = self.conv3D21(inp)
        x2_2 = self.conv3D22(x2_1)
        x2_3 = self.conv3D23(x2_2)

        fused_3 = torch.cat((x1_6, x2_3), dim=2)
        x_3 = self.conv3D31(fused_3)

        x_3 = x_3.squeeze(dim=2)

        x = self.conv2d_1(x_3)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)

        out = self.conv2d_4(x)

        return out
