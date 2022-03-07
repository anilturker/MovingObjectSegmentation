import torch
import torch.nn as nn

class AvFeat(nn.Module):

    def __init__(self, temporal_length, filter_size, activation=nn.ReLU()):
        super().__init__()
        self.conv1_5x5 = nn.Conv3d(1, filter_size, kernel_size=(3, 5, 5), stride=(5, 1, 1), padding=(0, 2, 2))
        self.conv1_3x3 = nn.Conv3d(1, filter_size, kernel_size=(3, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1))
        self.conv1_1x1 = nn.Conv3d(1, filter_size, kernel_size=(3, 1, 1), stride=(5, 1, 1), padding=0)

        self.conv1_avg_1x1 = nn.Conv3d(filter_size * 3, filter_size, 1, stride=(1, 1, 1), padding=0)

        self.conv2_5x5 = nn.Conv3d(filter_size, filter_size, kernel_size=(3, 5, 5), stride=(5, 1, 1), padding=(0, 2, 2))
        self.conv2_3x3 = nn.Conv3d(filter_size, filter_size, kernel_size=(3, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1))
        self.conv2_1x1 = nn.Conv3d(filter_size, filter_size, kernel_size=(3, 1, 1), stride=(5, 1, 1), padding=0)

        self.conv2_avg_1x1 = nn.Conv3d(filter_size * 3, filter_size, 1, stride=(1, 1, 1), padding=0)

        self.conv3_5x5 = nn.Conv3d(filter_size, filter_size, kernel_size=(3, 5, 5), stride=(2, 1, 1), padding=(1, 2, 2))
        self.conv3_3x3 = nn.Conv3d(filter_size, filter_size, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))
        self.conv3_1x1 = nn.Conv3d(filter_size, filter_size, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.conv3_avg_1x1 = nn.Conv3d(filter_size * 3, filter_size, 1, stride=(1, 1, 1), padding=0)

    def forward(self, inp):
        x1_1 = self.conv1_5x5(inp)
        x1_2 = self.conv1_3x3(inp)
        x1_3 = self.conv1_1x1(inp)

        fused1 = torch.cat((x1_1, x1_2, x1_3), dim=1)
        fused1 = self.conv1_avg_1x1(fused1)

        x2_1 = self.conv2_5x5(fused1)
        x2_2 = self.conv2_3x3(fused1)
        x2_3 = self.conv2_1x1(fused1)

        fused2 = torch.cat((x2_1, x2_2, x2_3), dim=1)
        fused2 = self.conv2_avg_1x1(fused2)

        x3_1 = self.conv3_5x5(fused2)
        x3_2 = self.conv3_3x3(fused2)
        x3_3 = self.conv3_1x1(fused2)

        fused3 = torch.cat((x3_1, x3_2, x3_3), dim=1)
        fused3 = self.conv3_avg_1x1(fused3)

        with torch.no_grad():
            out = fused3.squeeze(dim=2)

        return out
