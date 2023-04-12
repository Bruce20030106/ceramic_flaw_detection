from hfenet import HFENet
from evaluation import Bin_classification_cal
import utils
import logging
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import cv2
import time
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 解析命令行参数决定main函数做什么
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train or test the model.')

    parser.add_argument(
        "--train",
        action="store_true",
        help="Define if we wanna to train the net"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Define if we wanna to test the net"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Define if we wanna to caculate the FLOPs and Params of the net"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Define if we wanna to run a demo"
    )
    return parser.parse_args()


# 记录日志信息
def get_logger(log_path='log_path'):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    timer = time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
    txthandle = logging.FileHandler((log_path + '/' + timer + 'log.txt'))
    txthandle.setFormatter(formatter)
    logger.addHandler(txthandle)
    return logger


# 计算评估指标的函数。它接受三个参数：

def caculate(output, label, clear=False):
    # output：模型的输出
    # label：标签数据
    # clear：一个布尔值，指示是否清空统计结果
    cal = Bin_classification_cal(output, label, 0.5, clear)
    # 将输出值转换为二进制分类结果。caculate_total() 方法返回计算得到的统计结果。
    return cal.caculate_total()


# 用于删除模型文件 dir_list 的长度超过 count，则删除最旧的文件
def del_models(file_path, count=5):
    dir_list = os.listdir(file_path)
    if not dir_list:
        print('file_path is empty: ', file_path)
        return
    else:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        print('dir_list: ', dir_list)
        if len(dir_list) > 5:
            os.remove(file_path + '/' + dir_list[0])

        return dir_list


# 生成输入和标签的数据，用于模型的训练和测试。
def getInput_and_Label_generator(data_path):  # data_path: 输入图片和标签的路径
    img_Path = data_path + "/img"
    l = os.listdir(img_Path)  # 列出输入图片路径下的所有文件名称列表
    random.shuffle(l)  # 随机打乱文件名称列表
    for filename in l:  # 对文件列表中的每个文件名执行以下操作：
        img_name = img_Path + '/' + filename  # 获取当前图片的路径
        label_name = data_path + '/lab/' + filename.split('.')[0] + "_label.bmp"  # 获取当前图片对应的标签路径
        # 读取当前图片为灰度图像
        img = cv2.imread(img_name, 0)
        # 对当前图片进行直方图均衡化处理
        img_filters = cv2.equalizeHist(img)
        # 将当前图片和直方图均衡化处理后的图片相加
        img_add = cv2.add(img, img_filters)
        # 将当前图片、直方图均衡化处理后的图片和相加后的图片 按照通道合并为一张图片
        img_merge = cv2.merge([img, img_filters, img_add])

        lab = cv2.imread(label_name, 0)  # 读取当前图片的标签为灰度图像
        img = utils.cvTotensor_img(img_merge)  # 将当前图片转换为张量
        lab = utils.cvTotensor(utils.label_preprocess(lab))  # 将当前标签转换为张量，并对标签进行预处理

        # 将当前的图片和标签作为generator的一个输出，该输出会在下次调用时继续往下执行，直到所有的图片和标签都被处理完毕为止
        yield img, lab


# 生成输入和标签的数据，用于模型的验证。
def getInput_and_Label_generator_valid(data_path):
    img_Path = data_path + "/img"
    l = os.listdir(img_Path)

    for filename in l:
        img_name = img_Path + '/' + filename
        label_name = data_path + '/lab/' + filename.split('.')[0] + "_label.bmp"
        img = cv2.imread(img_name, 0)
        img_filters = cv2.equalizeHist(img)
        img_add = cv2.add(img, img_filters)
        img_merge = cv2.merge([img, img_filters, img_add])

        lab = cv2.imread(label_name, 0)
        img = utils.cvTotensor_img(img_merge)
        lab = utils.cvTotensor(utils.label_preprocess(lab))

        yield img, lab


# 定义变量并初始化HFENet模型
iterations = 0  # 设置迭代次数的初始值为0。
net = HFENet()  # 实例化一个HFENet模型。
net.cuda()  # 将模型移动到GPU上进行训练。

criterion = nn.BCELoss(weight=None, reduction='mean')  # 定义损失函数为二分类交叉熵损失函数。
optimizer = optim.Adam(net.parameters(), lr=0.01)  # 定义优化器为Adam，并设置学习率为0.01。
# 设置路径
valid_path = "./HFENet_Dataset/valid"
positive_path = "./HFENet_Dataset/train/defective"
negative_path = "./HFENet_Dataset/train/no_defective"
model_path = "./checkpoint"
log_path = "./log"


def train(net, epoch, iterations, loss_stop, positive_path, negative_path):
    # 设置模型为训练模式
    net.train()
    # 初始化该轮训练的总损失
    epoch_loss = 0.0
    # 打印训练开始信息
    print('train...')
    # 创建两个数据生成器，用于产生训练数据，一个产生正样本数据，一个产生负样本数据
    g_postive = getInput_and_Label_generator(positive_path)
    g_negative = getInput_and_Label_generator(negative_path)

    # 进行 iterations 次迭代
    for iters in tqdm(range(iterations)):
        # 每次循环，随机选择一个数据集进行训练，0表示正样本，1表示负样本
        for index in range(2):
            if index == 0:
                inputs, labels = next(g_postive)
            else:
                inputs, labels = next(g_negative)

            # 将输入和标签转移到GPU上
            inputs = inputs.cuda()
            labels = labels.cuda()

            # 将梯度归零，清空之前累积的梯度
            optimizer.zero_grad()
            # 用当前的网络对输入进行预测
            outputs = net(inputs)

            # 计算当前批次数据的损失
            lab = labels.detach().cpu().squeeze().numpy()
            out = outputs.detach().cpu().squeeze().numpy()
            loss = criterion(outputs, labels)

            # 反向传播，计算梯度
            loss.backward()
            # 更新权重
            optimizer.step()
            # 累加当前批次数据的损失
            epoch_loss += loss

    # 计算平均损失
    epoch_loss_mean = epoch_loss / iterations
    # 打印本轮训练的总损失和平均损失
    print('Train Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}'.format(epoch, epoch_loss.item(),
                                                                               epoch_loss_mean.item()))
    # 将训练信息写入日志文件中
    logger.info('Train Epoch:[{}] , loss: {:.6f}'.format(epoch, epoch_loss.item()))
    if epoch_loss < loss_stop:
        # 如果本轮训练的总损失小于预设的损失阈值，则返回True和本轮训练的总损失
        return True, epoch_loss
    else:
        # 否则，返回False和本轮训练的总损失
        return False, epoch_loss

def valid(net, epoch, img_path):
	#net.eval()
	valid_loss = 0.0
	img_Path = img_path + "/img"
	l = os.listdir(img_Path)
	iterations = len(l)
	print('img_Path: ', img_Path, 'len: ', iterations)
	g_data = getInput_and_Label_generator_valid(img_path)
	IoU1 = 0
	IoU2 = 0
	MIoU = 0
	PA = 0

	total_time = 0
	with torch.no_grad():
		# for iters in tqdm(range(iterations)):
		for iters in range(iterations):
			# 从生成器中获取输入数据和标签
			inputs, labels = next(g_data)
			# 将输入数据和标签移动到GPU上
			inputs = inputs.cuda()
			labels = labels.cuda()

			optimizer.zero_grad()
			# 用时计算检测时间
			torch.cuda.synchronize()
			begin_time = time.perf_counter()
			# 模型前向计算
			outputs = net(inputs)
			# 用时计算检测时间
			torch.cuda.synchronize()
			end_time = time.perf_counter()
			# 计算检测时间
			interval_time = end_time - begin_time
			total_time += interval_time
			print('detect time:', interval_time,'s')
			# 将张量转换为numpy数组
			lab = labels.detach().cpu().squeeze().numpy()
			out = outputs.detach().cpu().squeeze().numpy()
			# 计算评估指标
			PA, FP, FN = caculate(out, lab, (not bool(iters)))
			# 可视化结果
			strIndex = str(epoch) + '_valid_' + str(iters)
			utils.visualization([inputs, labels, outputs], strIndex)
			# 计算总的损失
			valid_loss += criterion(outputs, labels)

		print('average detect time:', total_time/iterations,'s')
		# 计算平均损失
		valid_loss_mean = valid_loss / iterations
		print('           Valid Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}\t FP: {}\t FN: {}\t FN+FP: {}\t PA: {:.6f}'.format(epoch, valid_loss.item(), valid_loss_mean.item(), FP, FN, FN+FP, PA))
		logger.info('         Valid Epoch:[{}] , loss: {:.6f}, FP: {}\t FN: {}\t FN+FP: {}\t PA: {:.6f}'.format(epoch, valid_loss.item(), FP, FN, FN+FP, PA))






def main(mode, epochs=100):
	# 打印网络结构
    print(net)

	# 获取数据集的路径和长度
    img_Path = positive_path + "/img"
    l = os.listdir(img_Path)
    iterations = len(l)
    # print('img_Path: ', img_Path, 'iterations: ', iterations)

	# 加载最新的模型
    if os.path.exists(model_path):
        dir_list = os.listdir(model_path)
        if len(dir_list) > 0:
            dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
            print('dir_list: ', dir_list)
            last_model_name = model_path + '/' + dir_list[-1]

			# 计算FLOPS
            if mode == 0:
                utils.calFlop(model=HFENet(), path=last_model_name)
                return

			# 加载模型
            checkpoint = torch.load(last_model_name)
            net.load_state_dict(checkpoint['model'])
            # params = net.state_dict().keys()
            # for i, j in enumerate(params):
            # 	print(i, j)
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print('load epoch {} succeed! loss: {:.6f} '.format(last_epoch, loss))
        else:
            last_epoch = 0
            print('no saved model')
    else:
        last_epoch = 0
        print('no saved model')


    if last_epoch == 0 and mode == -1:
        return

	# 根据mode变量决定训练或验证模型
    for epoch in range(last_epoch + 1, epochs + 1):
		# 训练模型
        if mode == 1:
            ret, loss = train(net=net, epoch=epoch, iterations=iterations, loss_stop=0.01, positive_path=positive_path,
                              negative_path=negative_path)
			# 保存模型
            state = {'model': net.state_dict(), 'epoch': epoch, 'loss': loss}
            model_name = model_path + '/model_epoch_' + str(epoch) + '.pth'
            torch.save(state, model_name)
        else:
			# 验证模型
            valid(net, epoch, valid_path)
            ret = True

		# 删除早期的模型
        # del_models(model_path)

		# 如果遇到 loss 低于loss_stop的限制，则停止训练
        if ret:
            break
	# 完成训练或验证，输出Done
    print('Done.')

# 加载最新的训练模型，并使用该模型对一张测试图像进行推理
def demo():
    if os.path.exists(model_path):  # 判断模型路径是否存在
        dir_list = os.listdir(model_path)   # 获取模型路径下的文件列表

        if len(dir_list) > 0:   # 如果模型文件存在
            dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))    # 按照文件修改时间排序
            print('dir_list: ', dir_list)
            last_model_name = model_path + '/' + dir_list[-1]   # 获取最新的模型文件
            checkpoint = torch.load(last_model_name)    # 获取最新的模型文件
            net.load_state_dict(checkpoint['model'])    # 加载模型的参数
            last_epoch = checkpoint['epoch']    # 获取最新的训练轮数
            loss = checkpoint['loss']   # 获取最新的训练损失
            print('load epoch {} succeed! loss: {:.6f} '.format(last_epoch, loss))
            print('start test...')
            print('load img...')
            img = cv2.imread('test.bmp', 0) # 加载模型的参数
            print('img preprocessing...')
            img_filters = cv2.equalizeHist(img) # 直方图均衡化
            img_add = cv2.add(img, img_filters) # 图像相加
            img_merge = cv2.merge([img, img_filters, img_add])  # 图像通道合并
            img = utils.cvTotensor_img(img_merge).cuda()    # 将图像转换为tensor并移动到GPU上
            print(img.shape)    # 模型推理
            print('inferencing...')
            outputs = net(img)
            print('inference done.')
            out = outputs.detach().cpu().squeeze().numpy() * 255    # 将输出转换为numpy数组并还原到0-255的范围
            print('save to result.bmp')
            cv2.imwrite('result.bmp', out)  # 保存推理结果
        else:
            print('no model!')

        sys.exit(0)


if __name__ == '__main__':
    args = parse_arguments()
    mode = 0

    if args.demo:
        demo()

    if args.train:
        mode = 1
    elif args.test:
        mode = -1
    elif args.info:
        mode = 0

    logger = get_logger(log_path)
    print('mode: ', mode)
    main(mode=mode)
