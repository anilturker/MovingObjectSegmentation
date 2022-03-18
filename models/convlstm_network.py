
# -*- coding: utf-8 -*-
"""
Akilan, Thangarajah @2018

This is a temporary script file.
"""

import torch
import torch.nn as nn


class Conv_block_3d(nn.Module):

	def __init__(self, ch_in, ch_out, batch_norm=False, activation=nn.ReLU(), kernel_size=(3, 3, 3),
				 stride = (1, 1, 1), padding = (0, 0, 0)):

		super(Conv_block_3d,self).__init__()
		self.conv3d = nn.Sequential()
		self.conv3d.add_module("conv3d", nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride,
												   padding=padding, bias=True))
		if batch_norm:
			self.conv3d.add_module("batchNorm3d", nn.BatchNorm3d(ch_out))
		if activation:
			self.conv3d.add_module("act", activation)

	def forward(self,x):
		x = self.conv3d(x)
		return x


class Sendec_block(nn.Module):

	def __init__(self, ch_in, ch_out):

		super(Sendec_block,self).__init__()
		self.conv3d_1 = Conv_block_3d(ch_in, 32, batch_norm=False, activation=nn.ReLU(),
									  kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
		self.conv3d_tranpose = nn.ConvTranspose3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
		ch_in = ch_in + 16
		self.bn = nn.BatchNorm3d(ch_in)

		self.conv3d_2 = Conv_block_3d(ch_in, ch_out, batch_norm=False, activation=nn.ReLU(), kernel_size=(1, 3, 3),
									  stride=(1, 1, 1), padding=(0, 1, 1))

	def forward(self, inp):

		x1 = self.conv3d_1(inp)
		x = self.conv3d_tranpose(x1)
		x = nn.functional.pad(x, (1, 0, 1, 0))
		x = torch.cat((inp, x), dim=1)
		x = self.bn(x)
		x = self.conv3d_2(x)
		return x1, x


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
			nn.BatchNorm2d(out_channels))

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


class SEnDec_cnn_lstm(nn.Module):

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

	def __init__(self, inp_ch=1):
		super(SEnDec_cnn_lstm, self).__init__()


		self.seq0 = Conv_block_3d(inp_ch, ch_out=16, batch_norm=True, activation=nn.ReLU(),
							 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

		# - SEnDec block 1
		self.seq1 = Sendec_block(16, ch_out=16)

		self.seq13 = Conv_block_3d(16, ch_out=32, batch_norm=True, activation=nn.ReLU(),
							 kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

		# - SEnDec block 2
		self.seq2 = Sendec_block(32, ch_out=16)

		self.seq22_conv = Conv_block_3d(16, ch_out=32, batch_norm=True, activation=nn.ReLU(),
							 kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

		# - SEnDec block 3
		self.seq3 = Sendec_block(32, ch_out=16)

		self.seq3_conv = Conv_block_3d(16, ch_out=32, batch_norm=True, activation=nn.ReLU(),
							 kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

		self.seq4 = ConvLSTMBlock(32, 16, kernel_size=3, padding=1)

		self.seq5 = Conv_block_3d(16, ch_out=16, batch_norm=True, activation=nn.ReLU(),
							 kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

		#-~~~~~~~~~~~~~~~~~~ Upsampling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		self.seq6_transpose = nn.ConvTranspose3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

		ch_in = 32 + 16
		self.seq6_conv = Conv_block_3d(ch_in, ch_out=32, batch_norm=True, activation=nn.ReLU(),
							 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

		self.seq7_transpose = nn.ConvTranspose3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

		ch_in = 32 + 16
		self.seq7_conv = Conv_block_3d(ch_in, ch_out=32, batch_norm=True, activation=nn.ReLU(),
							 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

		self.seq8_transpose = nn.ConvTranspose3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

		ch_in = 32 + 16
		self.seq8_conv = Conv_block_3d(ch_in, ch_out=32, batch_norm=True, activation=nn.ReLU(),
							 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

		self.seq9_transpose = nn.ConvTranspose3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

		ch_in = 16 + 16
		self.seq9_conv = Conv_block_3d(ch_in, ch_out=32, batch_norm=True, activation=nn.ReLU(),
							 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

		self.seq10 = ConvLSTMBlock(32, 16, kernel_size=3, padding=1)

		self.out = Conv_block_3d(16, ch_out=1, batch_norm=True, activation=nn.Sigmoid(),
							 kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

	def forward(self, inp):

		# 4D to 5D
		with torch.no_grad():
			inp = torch.tensor(inp, dtype=torch.float).unsqueeze(dim=1)

		seq0 = self.seq0(inp)
		seq1, seq12 = self.seq1(seq0)
		seq13 = self.seq13(seq12)

		seq2, seq22 = self.seq2(seq13)
		seq2_conv = self.seq22_conv(seq22)

		seq30, seq32 = self.seq3(seq2_conv)
		seq3_conv = self.seq3_conv(seq32)

		# ConvLSTM
		seq4 = self.seq4(seq3_conv)
		seq5 = self.seq5(seq4)

		# -~~~~~~~~~~~~~~~~~~ Upsampling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		seq6 = self.seq6_transpose(seq5)
		seq6 = nn.functional.pad(seq6, (1, 0, 1, 0))
		seq6 = torch.cat((seq6, seq30), dim=1)
		seq6 = self.seq6_conv(seq6)

		seq7 = self.seq7_transpose(seq6)
		seq7 = nn.functional.pad(seq7, (1, 0, 1, 0))
		seq7 = torch.cat((seq7, seq2), dim=1)
		seq7 = self.seq7_conv(seq7)

		seq8 = self.seq8_transpose(seq7)
		seq8 = nn.functional.pad(seq8, (1, 0, 1, 0))
		seq8 = torch.cat((seq8, seq13), dim=1)
		seq8 = self.seq8_conv(seq8)

		seq9 = self.seq9_transpose(seq8)
		seq9 = nn.functional.pad(seq9, (1, 0, 1, 0))
		seq9 = torch.cat((seq9, seq0), dim=1)
		seq9 = self.seq9_conv(seq9)

		seq10 = self.seq10(seq9)

		# Channel averaging
		avg = torch.mean(seq10, dim=2).unsqueeze(dim=2)

		out = self.out(avg)

		return out


if __name__ == '__main__':
	import time
	import torch
	from torch.autograd import Variable
	from torchsummaryX import summary

	torch.cuda.set_device(0)
	net =SEnDec_cnn_lstm(inp_ch=1).cuda().eval()

	data = Variable(torch.randn(1, 1, 16, 112, 112)).cuda()

	out = net(data)

	summary(net,data)
	print("out size: {}".format(out.size()))