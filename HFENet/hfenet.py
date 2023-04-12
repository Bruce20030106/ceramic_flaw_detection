########Architecture of our proposed model (HFENet)#########

import torch
import torch.nn as nn
import numpy as np


# 定义一个卷积层类，继承nn.Module
class convLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
		super(convLayer, self).__init__()

		# 定义激活函数ReLU
		self.activate = nn.ReLU(inplace=True)

		# 定义卷积层
		# in_channels和out_channels分别表示输入和输出的通道数；
		# kernel_size表示卷积核的大小；stride表示步长；padding表示填充数；dilation表示卷积核扩张率
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
							  stride=stride,
							  padding=padding, dilation=dilation, bias=False)

		# 定义InstanceNorm归一化层
		self.norm = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)

	def forward(self, x):
		# 前向传播
		x = self.conv(x)  # 卷积
		x = self.norm(x)  # 归一化
		x = self.activate(x)  # 激活函数
		return x	# 返回激活后的特征图


# 残差模块A
# 残差模块A的输入和输出具有相同的维度，主要由两个卷积层和两个归一化层组成，
# 其中一个卷积层采用较小的卷积核和较小的padding和dilation参数，（self.conv1使用的是5x5的正方形卷积核）
# 另一个卷积层采用更大的卷积核和更大的padding和dilation参数，以便增加网络的感受野和非线性能力。（self.conv2使用的是5x1的长方形卷积核）
# 模块中还包含激活函数ReLU。
class ResModule_A(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResModule_A, self).__init__()

		# 定义了激活函数ReLU和两个卷积层，其中卷积核的大小为5x5和5x1
		# 卷积核个数为out_channels，stride为1，padding为2和(4,0)，dilation为1和(2,1)
		# bias为False
		self.activate = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
							   dilation=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 1), stride=1,
							   padding=(4, 0), dilation=(2, 1), bias=False)

		# 定义了两个Instance Normalization层，num_features为out_channels，affine和track_running_stats为True和False
		self.norm1 = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)
		self.norm2 = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)

	def forward(self, x):
		# 定义了shortcut，即x经过一层卷积、Instance Normalization和ReLU后的输出
		shortcut = self.conv1(x)
		shortcut = self.norm1(shortcut)
		shortcut = self.activate(shortcut)
		shortcut = self.conv2(shortcut)
		shortcut = self.norm2(shortcut)

		# 输出为x和shortcut相加后再经过一层ReLU的结果
		return self.activate(x + shortcut)


# 残差块B
class ResModule_B(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResModule_B, self).__init__()
		# 定义组成 ResModule_B 的层
		self.activate = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5,1), stride=1, padding=(4,0), dilation=(2,1) , bias=False)
		self.norm1 = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)
		self.norm2 = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)

	def forward(self, x):
		# 建立 Residual Connection，并计算结果
		shortcut = self.conv1(x)
		shortcut = self.norm1(shortcut)
		shortcut = self.activate(shortcut)
		shortcut = self.conv2(shortcut)
		shortcut = self.norm2(shortcut)
		return self.activate(x+shortcut)

# Residual ModuleA中的第二个卷积层使用了不同的kernel_size和padding，这可以使卷积层沿水平方向具有更大的感受野，而不增加网络的参数量。
# 而Residual ModuleB中的第一个卷积层的kernel_size为3，没有使用padding，这样可以保持特征图的尺寸不变。


# 特征融合组件
class FeatureFusion(nn.Module):
	def __init__(self):
		super(FeatureFusion, self).__init__()
		# 定义Prewitt_Operator卷积层，用于提取竖直方向的边缘特征.
		# Prewitt_Operator 是指 Prewitt 算子，是一种常用的边缘检测算子。
		# 这里的 Prewitt_Operator 实际上是一个卷积层，其输入通道数为 3，输出通道数为 3，卷积核大小为 5，填充为 2，步长为 1，组数为 3，没有偏置。
		self.Prewitt_Operator = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2, dilation=1, groups=3, bias=False)

	def forward(self, x):
		# 使用Prewitt_Operator卷积层提取竖直方向的边缘特征
		vertical_edge = self.Prewitt_Operator(x)
		# 将原始输入x和竖直方向的边缘特征vertical_edge按通道维度拼接
		x = torch.cat([x, vertical_edge], dim=1)
		return x

# HFENet
class HFENet(nn.Module):
	def __init__(self):
		super(HFENet, self).__init__()

		# 定义一个特征融合组件
		self.featurefusion  = FeatureFusion()

		# 定义第一个5x5的单卷积层
		self.firstConvLayer = convLayer(in_channels=6, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1)

		# 定义一个ResModule_A残差模块
		self.resmoduleA = ResModule_A(in_channels=32, out_channels=32)

		# 定义六个ResModule_B残差模块
		self.resmoduleB1 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB2 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB3 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB4 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB5 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB6 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB7 = ResModule_B(in_channels=32, out_channels=32)

		# 定义三个上采样层（作用是将输入图像或特征图的尺寸放大，从而增加图像或特征图中的像素数量和细节信息。）
		self.upsample1_conv = convLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
		self.upsample2_conv = convLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
		self.upsample3_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

		# 定义最大池化层
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# 定义upsample3_conv的激活函数
		self.sigmoid = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data)
			elif isinstance(m, nn.InstanceNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

		# 定义Prewitt算子
		prewittWeight = np.array([
				   [2, 1, 0, -1, -2],
				   [2, 1, 0, -1, -2],
				   [2, 1, 0, -1, -2],
				   [2, 1, 0, -1, -2],
				   [2, 1, 0, -1, -2]], np.float32)

		# 将算子嵌入卷积核中
		# 输出旧的卷积核权重的形状
		print('shape of old weight : ', self.featurefusion.Prewitt_Operator.weight.shape)
		# 将预处理好的算子权重增加一个新轴，将其形状变为(1, 3, 3)，用于与卷积核的形状(64, 3, 3)相匹配
		prewittWeight = prewittWeight[np.newaxis, :]
		# 再增加一个新轴，将其形状变为(1, 1, 3, 3)，用于与卷积核的形状(64, 3, 3)相匹配
		prewittWeight = np.expand_dims(prewittWeight,0).repeat(3, axis=0)
		# 输出新的卷积核权重的形状
		print('shape of my weight: ', prewittWeight.shape)

		# 将numpy数组转换为PyTorch张量
		prewittWeight = torch.FloatTensor(prewittWeight)

		# 将卷积核权重参数设置为prewittWeight张量，requires_grad=False表示此参数不需要梯度更新
		self.featurefusion.Prewitt_Operator.weight = torch.nn.Parameter(data=prewittWeight, requires_grad=False)
		print('Operator embedding successful!')
		# 输出新的卷积核权重的形状
		print('shape of new weight: ', self.featurefusion.Prewitt_Operator.weight.shape)
		# 输出新的卷积核权重参数的值
		print(self.featurefusion.Prewitt_Operator.weight)

	def forward(self, x):
		# 特征融合层
		# 特征融合的作用是将来自不同卷积层的特征进行融合，以提高特征的表征能力。
		# 卷积神经网络通过层叠多个卷积层，可以提取出不同层次的特征信息。
		# 通过特征融合将这些不同层次的特征信息进行整合，从而提高模型对图像的表征能力。

		x = self.featurefusion(x)
		# 第一个卷积层
		x = self.firstConvLayer(x)
		# 备份 x
		x_bk1 = x

		# max pooling 操作
		x = self.pool(x)
		# 残差块 A
		x = self.resmoduleA(x)
		# 备份 x
		x_bk2 = x

		# max pooling 操作
		x = self.pool(x)
		# 残差块 B1-B3
		x = self.resmoduleB1(x)
		x = self.resmoduleB2(x)
		x = self.resmoduleB3(x)
		# 备份 x
		x_bk3 = x

		# 以上备份的作用是：前面的模块的输出将会在后面的模块中使用，因此需要将前面的输出备份，以便在需要的时候使用。

		# max pooling 操作
		x = self.pool(x)
		# 残差块 B4-B6
		x = self.resmoduleB4(x)
		x = self.resmoduleB5(x)
		x = self.resmoduleB6(x)

		# 上采样操作
		x = self.upsample(x)
		# 将 x_bk3 与上采样后的结果拼接
		x = torch.cat([x, x_bk3], dim=1)
		x = self.upsample1_conv(x)

		# 再次上采样操作
		x = self.upsample(x)
		# 将 x_bk2 与上采样后的结果拼接
		x = torch.cat([x, x_bk2], dim=1)
		x = self.upsample2_conv(x)

		# 再次上采样操作
		x = self.upsample(x)
		# 将 x_bk1 与上采样后的结果拼接
		x = torch.cat([x, x_bk1], dim=1)
		x = self.upsample3_conv(x)

		# 使用 sigmoid 函数进行激活
		x = self.sigmoid(x)

		# 在进行完所有的特征提取之后，
		# 执行一系列的上采样、特征融合和卷积操作，将特征图恢复到原始大小，通过 sigmoid 函数进行激活。
		# 最终，输出的张量 x 是一个概率图，用于表示输入图像中每个像素为背景或前景（分割目标）的概率。

		return x


if __name__ == '__main__':
	hfenet = HFENet()
	print(hfenet)

