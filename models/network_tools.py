import torch
import torch.nn as nn


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
    def __init__(self, ch_in, ch_out, batch_norm=True, activation=nn.ReLU(), kernel_size=(3, 3, 3), stride=(1, 1, 1),
                 dilation=(1,1,1), padding=(0, 0, 0)):
        super(conv_block_3d,self).__init__()
        self.conv3d = nn.Sequential()
        self.conv3d.add_module("conv3d", nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride,
                                                   padding=padding, bias=True))
        if batch_norm:
            self.conv3d.add_module("batchNorm3d", nn.BatchNorm3d(ch_out))

        self.conv3d.add_module("act", activation)

    def forward(self,x):
        x = self.conv3d(x)
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


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=2, padding=0, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=1), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.up(x)
        return x


class ConvLSTMBlock(nn.Module):

    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, inputs):
        '''
        :param inputs: (B, C, S, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, C, S, H, W = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4)  # (b, c, s, h, w) -> (b, s, c, h, w)
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 2, 0, 3, 4).contiguous() # (S, B, C, H, W) -> (B, C, S, H, W)


class UNetDown(nn.Module):
    """ Encoder blocks of UNet

    Args:
        in_ch (int): Number of input channels for each conv layer
        out_ch (int): Number of output channels for each conv layer
        num_rep (int): Number of repeated conv-batchnorm layers
        batch_norm (bool): Whether to use batch norm after conv layers
        activation (torch.nn module): Activation function to be used after each conv layer
        kernel_size (int): Size of the convolutional kernels
        dropout (booelan): Whether to apply spatial dropout at the end
        maxpool (booelan): Whether to apply max pool in the beginning
    """
    def __init__(self, in_ch, out_ch, num_rep, batch_norm=False, activation=nn.ReLU(), kernel_size=3,
                 dropout=False, maxpool=False):
        super().__init__()
        self.down_block = nn.Sequential()

        if maxpool:
            self.down_block.add_module("maxpool", nn.MaxPool2d(2))
        in_ch_for_conv = in_ch
        for k in range(num_rep):
            self.down_block.add_module("conv%d"%(k+1), nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, padding=(int((kernel_size-1)/2))))
            self.down_block.add_module("act%d"%(k+1), activation)
            if batch_norm:
                self.down_block.add_module("bn%d"%(k+1), nn.BatchNorm2d(out_ch))
            in_ch_for_conv = out_ch
        if dropout:
            self.down_block.add_module("dropout", nn.Dropout2d(p=0.5))

    def forward(self, inp):
        return self.down_block(inp)


class UNetUp(nn.Module):
    """ Decoder blocks of UNet

    Args:
        in_ch (int): Number of input channels for each conv layer
        res_ch (int): Number of channels coming from the residual, if equal to 0 and no skip connections
        out_ch (int): Number of output channels for each conv layer
        num_rep (int): Number of repeated conv-batchnorm layers
        batch_norm (bool): Whether to use batch norm after conv layers
        activation (torch.nn module): Activation function to be used after each conv layer
        kernel_size (int): Size of the convolutional kernels
        dropout (booelan): Whether to apply spatial dropout at the end
    """

    def __init__(self, in_ch, res_ch, out_ch, num_rep, batch_norm=False, activation=nn.ReLU(), kernel_size=3,
                 dropout=False):

        super().__init__()
        self.up = nn.Sequential()
        self.conv_block = nn.Sequential()

        self.up.add_module("conv2d_transpose", nn.ConvTranspose2d(in_ch, in_ch, kernel_size, stride=2,
                                                                  output_padding=(int((kernel_size-1)/2)),
                                                                  padding=(int((kernel_size-1)/2))))
        if batch_norm:
            self.up.add_module("bn1", nn.BatchNorm2d(in_ch))

        in_ch_for_conv = in_ch + res_ch
        for k in range(num_rep):
            self.conv_block.add_module("conv%d"%(k+1), nn.Conv2d(in_ch_for_conv, out_ch,
                                                        kernel_size=kernel_size, padding=(int((kernel_size-1)/2))))
            self.conv_block.add_module("act%d"%(k+1), activation)
            if batch_norm:
                self.conv_block.add_module("bn%d"%(k+2), nn.BatchNorm2d(out_ch))
            in_ch_for_conv = out_ch
        if dropout:
            self.conv_block.add_module("dropout", nn.Dropout2d(p=0.5))

    def forward(self, inp, res=None):
        """
        Args:
            inp (tensor): Input tensor
            res (tensor): Residual tensor to be merged, if res=None no skip connections
        """
        feat = self.up(inp)
        if res is None:
            merged = feat
        else:
            merged = torch.cat([feat, res], dim=1)
        return self.conv_block(merged)


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


class FCNN(nn.Module):
    """ Fully connected Neural Network with Softmax in the end

        Args:
            sizes ([int]): Sizes of the layers starting from input till the output
    """

    def __init__(self, sizes):
        super().__init__()
        self.fcnn = nn.Sequential()
        for k, (in_ch, out_ch) in enumerate(zip(sizes[:-2], sizes[1:-1])):
            self.fcnn.add_module("fc%d" %(k+1), nn.Linear(in_ch, out_ch))
            self.fcnn.add_module("bn%d" %(k+1), nn.BatchNorm1d(out_ch))
            self.fcnn.add_module("relu%d" %(k+1), nn.ReLU(True))
        self.fcnn.add_module("fc%d" %(len(sizes)-1), nn.Linear(sizes[-2], sizes[-1]))
        self.fcnn.add_module('softmax', nn.LogSoftmax(dim=1))

    def forward(self, inp):
        return self.fcnn(inp)
