import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchstat import stat
from thop import profile


# 拼接并保存原始图像、标签和结果图像
def visualization(tensor, strIndex):
    """
	将传入的张量中的三个元素分别作为输入图像、标签和输出图像，
	可视化显示在一张图像上，并将结果保存到指定路径下的图像文件中。
	Args:
	- tensor: 一个形状为 [3,1,H,W] 的张量，其中三个元素分别为输入图像、标签和输出图像
	- strIndex: 一个字符串类型的变量，表示将要保存的图像的文件名
	Returns:
	- None
	"""

    img = tensor[0][0]  # 获取输入图像
    lab = tensor[1][0]  # 获取标签
    out = tensor[2][0]  # 获取输出图像

    # 将图像从张量中取出来，并将其转换为 numpy 数组
    img = img.detach().cpu().squeeze().numpy()
    lab = lab.detach().cpu().squeeze().numpy()
    out = out.detach().cpu().squeeze().numpy()

    # 将三个图像可视化在一张图像中，并将结果保存到指定路径下的图像文件中
    plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title('Input')
    plt.imshow(img[0], cmap="gray")
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title('Label')
    plt.imshow(lab, cmap="gray")
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title('Output')
    plt.imshow(out, cmap="gray")

    picName = './visualization/' + strIndex + '.jpg'
    plt.savefig(picName)
    plt.cla()
    plt.close("all")


def ImageBinarization(img, threshold=1):
    """
	将输入的图像二值化处理，并返回二值化后的结果。
	Args:
	- img: 一个 numpy 数组，表示需要进行二值化处理的图像
	- threshold: 一个整数，表示二值化阈值
	Returns:
	- image: 一个 numpy 数组，表示二值化处理后的图像
	"""
    img = np.array(img)
    image = np.where(img > threshold, 1, 0)
    return image


def label_preprocess(label):
    """
	对标签图像进行预处理，将其进行二值化处理。
	Args:
	- label: 一个 numpy 数组，表示需要进行预处理的标签图像
	Returns:
	- label_pixel: 一个 numpy 数组，表示经过预处理后的标签图像
	"""
    label_pixel = ImageBinarization(label)
    return label_pixel


def cvTotensor(img):
    """
    功能：将numpy数组的图像转换为PyTorch张量
    参数：
        img：numpy数组的图像
    返回值：
        tensor：PyTorch张量的图像
    """
    # 在图像最后一维上增加一个维度，变成[H,W,C]->[H,W,C,1]
    img = (np.array(img[:, :, np.newaxis]))
    # 将图像的维度顺序从[H,W,C,1]转换为[1,C,H,W]
    img = np.transpose(img, (2, 0, 1))
    # 在第一个维度上增加一个维度，变成[1,C,H,W]->[1,1,C,H,W]
    img = (np.array(img[np.newaxis, :, :, :]))
    # 将numpy数组转换为PyTorch张量
    tensor = torch.from_numpy(img)
    # 将张量的数据类型转换为float32
    tensor = torch.as_tensor(tensor, dtype=torch.float32)
    return tensor


def cvTotensor_img(img):
    """
    功能：将numpy数组的图像转换为PyTorch张量
    参数：
        img：numpy数组的图像
    返回值：
        tensor：PyTorch张量的图像
    """
    # 将图像的维度顺序从[H,W,C]转换为[C,H,W]
    img = np.transpose(img, (2, 0, 1))
    # 在第一个维度上增加一个维度，变成[1,C,H,W]
    img = (np.array(img[np.newaxis, :, :, :]))
    # 将numpy数组转换为PyTorch张量
    tensor = torch.from_numpy(img)
    # 将张量的数据类型转换为float32
    tensor = torch.as_tensor(tensor, dtype=torch.float32)
    return tensor


def caculate_FLOPs_and_Params(model):
    """
    功能：计算模型的FLOPs和参数数量
    参数：
        model：需要计算FLOPs和参数数量的模型
    返回值：
        flops：模型的FLOPs数量
        params：模型的参数数量
    """
    # 生成一个1x3x1408x256的随机输入张量
    input = torch.randn(1, 3, 1408, 256)
    # 使用thop库计算模型的FLOPs和参数数量
    flops, params = profile(model, inputs=(input,))
    # 打印模型的FLOPs和参数数量
    print('flops: ', flops, ' params: ', params)
    return flops, params


def calFlop(model, path):
    # 加载指定路径下的模型权重
    checkpoint = torch.load(path, map_location='cpu')
    # 将指定模型的权重加载到模型中
    model.load_state_dict(checkpoint['model'])
    # 使用thop库计算指定模型的FLOPs和参数数量
    stat(model, (3, 1408, 256))
