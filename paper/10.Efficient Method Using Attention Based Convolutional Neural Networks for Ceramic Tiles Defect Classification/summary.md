# 一、简介：
在瓷砖的生产阶段，不同的缺陷会影响瓷砖表面的质量，如裂纹、斑点、针孔、和拐角或边缘断裂。因此，要实现瓷砖质量检测系统的完全自动化，还必须解决其他问题，尤其是形状和纹理缺陷的多样性。传统技术的特点是所提出的系统**缺乏稳健性**和**生产性能降低**。

基于卷积神经网络络的技术丢失空间信息，阻碍了有效的局部和全局特征提取，以
在瓷砖缺陷检测的背景下解决这个问题。模型基于卷积架构，并使用**卷积块注意模块(CBAM)**来更多地关注输入图像的相关区域，以**克服空间信息丢失问题**。在训练之前，进行预处理步骤，通过使用**适当的掩码处理**与缺陷类型相对应的每个图像来促进学习。实验结果表明，作者的模型在减少参数的同时，能够准确高效地分类陶瓷瓷砖缺陷。作者还提出了一种新的瓷砖缺陷数据集，该数据集来自一个陶瓷生产单位。实验结果表明，与最先进的方法相比，**作者提出的方法平均精度率为99.93%**。模型考虑的表面缺陷是角断裂、针孔和图案不连续性。
## 掩码（mask image）：
mask image（掩模图像）是一种二进制图像，其中包含了用于指示其他图像中某些位置信息的图像。通常，掩模图像中的像素值为**1的位置表示了需要注意的区域**，而**像素值为0的位置则表示不需要注意的区域**。掩模图像可以用于许多任务，例如物体分割、目标检测和图像增强等。

	如图
![](https://github.com/dhf97/image/raw/main/maskImage.png)

## 空间信息丢失问题：
在卷积神经网络中，每个卷积层都会将输入图像进行卷积操作，从而提取图像中的特征。然而，随着网络的深入，**特征图的大小会逐渐缩小**，这可能会导致一些重要的空间信息丢失，例如物体的边缘和细节。**空间信息丢失问题就是指由于这种特征图大小的缩小而导致的空间信息的丢失。**
### 卷积操作的过程：
1. **定义卷积核**（或称为滤波器或过滤器）的**大小和数量**。卷积核通常是一个小的二维矩阵，其大小和数量可以根据需求自由设置。
2. 将**卷积核**从输入数据的左上角开始，**按照步长（stride）的大小进行滑动**。步长定义了卷积核每次在输入数据中移动的距离。
3. 将卷积核覆盖在当前位置的输入数据上，计算卷积核中的元素与输入数据中对应位置的元素的乘积，并将它们相加**得到**一个标量值。这个标量值就是卷积操作的**输出值**。
4. 将卷积核在输入数据上继续滑动，直到遍历完整个输入数据。每个卷积核都会生成一个对应的二维矩阵，这个矩阵就是**卷积操作的输出结果，也称为特征图**。
5. **将特征图作为下一层的输入**，并重复上述过程，直到得到最终的输出结果。

卷积操作的目的是**提取输入数据中的局部特征**，因此卷积核的大小和数量以及步长的设置都会影响卷积操作的输出结果。在训练卷积神经网络时，卷积核的**权重会随着反向传播算法的迭代而不断优化**，使得网络能够自动学习到输入数据中的特征，从而实现对输入数据的分类、识别、检测等任务。

# 二、相关工作：
	机翻：尽管强烈依赖于光照、圆盘晶粒结构、缺陷形态和缺陷尺寸 [2]，但传统的缺陷检测算法并未取得稳定的结果。另一方面，已经提出了几项研究来实现瓷砖缺陷检测的高精度 [3-9]。Ragab 和 Alsharay [3] 提出了一个由两个步骤组成的解决方案，以最大限度地减少检测陶瓷图像中的缺陷所需的时间。第一步，它检测图像八个部分中的每一个部分的缺陷，第二步，它使用斑点和裂纹缺陷算法确定它们的类型。实验表明，与以前的工作相比，检测和分类瓷砖缺陷的时间减少了。然而，它们使用的图像数量有限，并且检测率太低。Hanzaei 等人。[4]采用中值滤波、局部方差旋转不变测度、阈值化和形态学闭包运算等处理技术，然后使用支持向量机（SVM）进行多类分类。该解决方案通过准确检测边缘和缺陷区域并剔除不重要区域，提高了瓷砖表面裂纹和孔洞的自动检测率。然而，这项工作不涉及瓷砖的纹理表面。而陈等人。[5] 使用决策树算法对纹理瓷砖的缺陷进行分类，基于傅里叶变换去除背景，他们应用拉普拉斯锐化、直方图规范和中值滤波。然后，他们使用 Lavel 共现矩阵 (GLCM) 和 Casagrande 等人提取特征。[6] 提出了一种从光滑和纹理瓷砖图像中提取特征的混合算法，在应用预处理技术后，使用基于分割的分形纹理分析和离散小波变换方法的组合，然后将其引入五个分类器。其中，SVM 分类器对平滑图像达到 99.01%，对纹理图像达到 97.89%。尽管使用纹理瓷砖，该解决方案仅处理图案不连续的缺陷。此外，它仅限于纹理瓷砖图案的单一模型。马里亚迪等人。[7] 提出使用 GLCM 提取缺陷区域的十四个属性，然后将其发送到人工神经网络进行分类。在计算平均色调、偏差平方和欧氏距离、阈值分割和形态学操作后，从图像中裁剪出缺陷区域。尽管没有纹理的图像数量有限，但缺陷分类存在错误，尤其是针孔被归类为裂纹。佐里奇等人。[8] 开发了一种检测饼干砖表面缺陷的方法，该方法基于傅里叶谱提取缺陷特征，并基于 K 最近邻 (KNN) 和随机森林进行分类。张等。[9] 提出了一种基于三个基本步骤的有效方法：图像预处理、缺陷检测和缺陷确定。作者通过SRR算法去除背景信息，调整横向对比度，然后将空间分布方差 (CSDA) 和色点区域权重 (CSAW) 应用于 HSV 颜色。生成缺陷显着图，并根据缺陷矩形的边界将缺陷区域分成块，提取特征向量颜色，然后将其馈送到SVM算法。该方法取得了良好的性能，准确率为98.75%。由于要表示的颜色特征数量的增加，该方法的性能随着瓷砖表面纹理的复杂性和多样性的增加而降低。然而，最近出现了基于深度学习模型的贡献。提出了一种新方法“陶瓷裂纹分割模型 (CCS)”[10]。的确，后者基于 U-net 模型和应用于图像的预处理技术的组合，以基于形态学和校正操作在白色和完美的字体中呈现突出的裂缝。缺陷检测的准确率非常高，达到99.9%。然而，该解决方案的有效性已经在有限数量的瓷砖图像上进行了测试，并且不会只处理一个缺陷。诺盖等人。[11] 使用 AlexNet 预训练模型根据声学噪声对瓷砖图像中的不可见裂缝进行分类。实际上，引入了九类大小不同但结构特性相似的不可见裂纹图像的数据集。该模型在检测裂缝方面取得了更好的准确性。尽管这项工作支持泛化，但该模型在检测裂缝方面取得了更好的准确性。然而，它没有指定裂纹的位置，而裂纹的位置对于裂纹的自动检查是必不可少的，并且它不会消除图像中的噪声。因此，有限数量的图像用于训练和测试。而斯蒂芬等人。[12] 提出了一种轻量级的卷积神经网络来自动检测具有光滑表面的瓷砖中的裂缝。在此 CNN 模型中并行执行特征提取和分类。确实获得了检测裂纹的更好准确度。

相关工作部分讨论了已提出的用于瓷砖缺陷检测的各种传统方法和基于深度学习的方法。**传统的缺陷检测算法由于对光照、圆盘晶粒结构、缺陷形貌和缺陷尺寸的强烈依赖，未能取得稳定的结果。**已经提出了几项研究来实现瓷砖缺陷检测的高精度，例如使用中值滤波器、局部方差旋转不变度量、阈值处理、形态闭包操作和支持向量机等处理技术进行多类分类。

基于深度学习的方法也已经出现，例如基于 U-net 模型的“陶瓷裂纹分割模型（CCS）”和预处理技术的组合，以白色和完美的字体渲染突出的裂纹。另一种方法使用 AlexNet 预训练模型根据声学噪声对瓷砖图像中的不可见裂缝进行分类。

尽管这些方法表现出良好的性能，但它们在用于训练和测试的图像数量以及它们可以检测到的缺陷类型方面受到限制。例如，CCS 模型仅检测裂缝并且已经在有限数量的瓷砖图像上进行了测试。同样，AlexNet 预训练模型不指定裂缝的位置，这对于自动化检查裂缝至关重要。

总的来说，仍然需要更准确、更稳健的缺陷检测方法来处理瓷砖上的各种缺陷和纹理。

# 三、相关技术：

## CBAM：
[CBAM详解](https://blog.csdn.net/Roaddd/article/details/114646354 "CBAM详解")

**Convolutional Block Attention Module**，是一种用于增强卷积神经网络（CNN）的**注意力机制**。It consists of two attention modules to allow the network to extract the relevant features by focusing on **what to pay attention by the Channel module** and **where to pay attention by the Spatial module**. 

	结构如图
![](https://github.com/dhf97/image/raw/main/CBAM.png)

CBAM 的主要功能是生成一个注意特征图（AttenF），该特征图从卷积特征图（ConvF）作为输入接受并与通道特征图和空间特征图相乘。如公式（1）所示：

AttenF = SAM(CAM(ConvF)ConvF).(CAM(ConvF) × ConvF) (1)

.在矩阵运算中代表内积（对应元素相乘）

具体来说，CBAM模块由Channel Attention Module (CAM)和Spatial Attention Module (SAM)组成，其中CAM和SAM分别用于提取输入特征图的通道注意力和空间注意力信息。在计算过程中，首先利用CAM生成**通道注意力特征图(CAM(ConvF))**，然后将其与输入特征图(ConvF)相乘得到**通道注意力特征图(ConvF)×CAM(ConvF)**。（CAM(ConvF)是由通道注意力模块生成的通道特征图，它被用来对输入的卷积特征图（ConvF）进行加权。通过乘上ConvF，CBAM可以获得空间信息和通道信息之间的相互作用，以更好地捕捉图像中的特征。）接着，利用SAM生成空间**注意力特征图(SAM(CAM(ConvF)ConvF))**，它将通道注意力特征图(ConvF)×CAM(ConvF)作为输入，并在空间维度上进行池化和卷积操作。最后将得到的空间注意力特征图(SAM(CAM(ConvF)ConvF))与通道注意力特征图相乘得到**最终的注意力特征图(AttenF)**。

## 通道注意力模块（CAM）
CAM 包括并行的一个最大池化（MaxP）层和一个平均池化（AvgP）层，接着应用多层感知机（MLP）和 sigmoid 函数。首先通过**全局池化操作**将每个通道的特征图转换为一个单一的标量值。这些标量值表示了每个通道对输入图像的重要性（**值越大表示该通道对于特征提取越重要**）。然后，这些标量值被送入两个全连接层，其中第一个全连接层（MLP）用于降低维度，第二个全连接层(MLP)用于增加维度，**最终得到一个与输入通道数相同的向量。该向量包含了每个通道的权重**，可以用于对输入特征图进行**加权求和**（见下图dtdjj1），以强化那些被认为重要的特征。
CAM(ConvF) = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 (𝑀𝐿𝑃 (𝐴𝑣𝑔𝑃(ConvF) + 𝑀𝐿𝑃(𝑀𝑎𝑥𝑃(ConvF))))(2)

	CAM如图
![](https://github.com/dhf97/image/raw/main/CAM.png)

### 通道：
在卷积神经网络中，通道是指**卷积核中的一个维度**，也可以称作卷积核的深度。每个卷积核都由多个通道组成，每个通道学习一个不同的特征。例如，一组由32个卷积核构成的卷积层，每个卷积核大小为3x3x3，其中**3代表了通道的数量**。这意味着在这个卷积层中，每个输入特征图都会被32个卷积核进行卷积操作，**每个卷积核中都包含了3个通道**，即32个卷积核中的每个通道都学习了不同的特征。最终，卷积操作的结果被沿着通道维度进行堆叠，生成新的特征图。通道的数量可以决定神经网络的复杂度和表达能力。

	如图
![](https://github.com/dhf97/image/raw/main/DTDJJ1.png)
![](https://github.com/dhf97/image/raw/main/DTDJJ11.png)
![](https://github.com/dhf97/image/raw/main/DTDJJ2.png)

### 全局池化操作：
对整个**特征图**进行池化，从而得到一个全局性的特征描述。常见的全局池化操作包括**全局平均池化（global average pooling）**和**全局最大池化（global max pooling）**。全局池化操作通常用于减少特征图的维度，并且可以使模型更加鲁棒，减少过拟合的风险。在注意力机制中，**全局池化操作可以用来计算每个通道的重要性，以便在不同的通道之间共享特征信息**。
#### 特征图（feature map）：
是指在每一层**卷积操作**中生成的一组**二维矩阵**。每个特征图表示了一个不同的抽象特征，这些特征可以用来识别输入图像中的不同部分。在卷积操作中，每个卷积核扫过输入图像，产生一组特征图，每个特征图对应着一个卷积核的输出结果。这些特征图经过激活函数的激活之后，作为下一层的输入，再次被卷积核扫过生成新的特征图，这个过程一直持续到最后一层，生成最终的输出结果。特征图中的每个元素表示了一个像素点，每个元素的大小和位置对应着输入图像中的一个局部区域。通过对特征图的操作，我们可以学习到输入图像中的不同抽象特征，从而实现对输入图像的分类、识别、检测等任务。
	
## 空间注意力模块（SAM）
SAM 按顺序包含一个最大池化层和一个平均池化层，然后是一个大小为 7×7 (Convf) 的单个卷积层和一个 sigmoid 函数。首先在通道维度平均池化和最大池化，然后将他们产生的特征图进行**拼接**起来（concat(在通道维度上进行拼接)）。然后在拼接后的特征图上，使用卷积操作来产生最终的空间注意力特征图。公式如下：

SAM(CAM(ConvF)ConvF) = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 (Convf 7×7 ([𝐴𝑣𝑔𝑃(CAM(ConvF)ConvF);
𝑀𝑎𝑥𝑃(CAM(ConvF)ConvF)]))


	SAM如图
![](https://github.com/dhf97/image/raw/main/SAM.png)

# 四、方法和材料：
1. 数据收集：收集有缺陷和无缺陷的瓷砖图像
2. 预处理：删除背景并为有缺陷的图像生成掩码
3. 分类：使用卷积神经网络进行分类，在 CNN 中引入了两层注意力机制。后者将
RGB 图像作为输入并产生有缺陷或无缺陷的二元分类。

	全局结构
![](https://github.com/dhf97/image/raw/main/GAS.png)

## 采集图像：

![](https://github.com/dhf97/image/raw/main/EOIFOD.png)

## 预处理：
第一阶段是从图像中**去除背景**，第二阶段是为有缺陷的图像**创建掩模**，以更好地表示缺陷的位置和形状，并减少在瓷砖上检测的复杂度。在工业检测系统的自动化中，检测纹理表面上不同类型的缺陷是一个巨大的挑战[23]。

陶瓷上每个缺陷的特殊性都需要不同的处理方法来制作掩模。添加这些掩码的目的是改善特征的表示。

### 缺角和针孔掩膜：

- 创建二值化图像
- 从灰度图像中，我们使用Otsu函数和反转操作来识别白色的瓷砖，黑色的背景和针孔

####Otsu函数：
[OTSU算法（大津法—最大类间方差法）原理及实现](https://blog.csdn.net/weixin_40647819/article/details/90179953)

该方法通过对图像的像素值进行聚类分析，自动确定分割阈值，将图像分为两个部分，从而实现图像分割的目的。Otsu函数的主要思想是寻找一个最优阈值，使得阈值分割后的两部分像素之间的类内方差最小，而类间方差最大。在具体实现上，Otsu函数通过计算图像的直方图来确定最优阈值，进而将图像进行二值化处理，**将亮度大于阈值的像素点设为前景（白色），将亮度小于阈值的像素点设为背景（黑色）。**

### 纹理遮罩：
为了更精确地表示瓷砖表面图案的不连续性，使用了Alvy Ray Smith创建的**HSV颜色模式**

## 分类：
分类模型采用了卷积神经网络（CNN）和注意力机制的结构。CNN由**四个卷积层**组成，前两个卷积层用于提取低层次特征，然后是ReLU激活函数和最大池化层。接着是CBAM层和两个卷积层，每层都应用ReLU函数。第二个CBAM注意力模块被插入。我们采用了**0.25的dropout因子进行正则化**，以避免过拟合。最后，采用100个神经元的全连接层（FC）和Sigmoid激活函数对输入图像进行分类，其尺寸为210×210×3，判断其是否有缺陷。

# 五、实验：

## 数据集和数据集扩充
我们的数据集包括两个标签，缺陷和非缺陷，用于进行二分类。缺陷类包括我们研究中考虑的三种缺陷的图像。应用数据增强操作以增加训练数据的样本数并减少过拟合的风险。对于**有缺陷的瓷砖图像，我们采用了180度旋转方法、水平和垂直翻转、缩放和剪切技术，范围为20%。同时，我们仅对正常图像应用2度旋转，以保持瓷砖的正确形状并避免破损。**表1描述了图像增强前后的图像总数。实际上，在预处理步骤中，我们添加了默认的图像掩码。我们从数据集中创建了三个图像分布来执行我们的实验，以评估我们模型的性能。分布的详细信息显示在表2中。**数据集1用于二分类，数据集2用于多类别分类，数据集3由没有掩码的图像组成。**

## 训练细节和评估指标

**学习率为1e-3，使用了RMSprop优化器，批量大小为15，输入图像大小设置为120×120**

### RMSprop优化器：
RMSprop（Root Mean Square Propagation）是一种常用的梯度下降算法的变体，它主要用于训练神经网络模型。与传统的梯度下降算法不同，RMSprop可以自适应地调整每个参数的学习率，以便更快地收敛。其基本思想是对梯度进行平方加权平均（类似于梯度下降算法中的动量），并使用该平均值来调整每个参数的学习率，以便能够快速收敛到最优值。此外，RMSprop还可以缓解梯度爆炸或消失的问题，从而提高模型的稳定性和收敛速度。

#### 梯度下降算法：
|:-------------:|:-------------:|:-------------:|
| 算法名称	|优点	|缺点|
|批量梯度下降（BGD）	|收敛性好	|慢，内存开销大|
|随机梯度下降（SGD）	|收敛速度快，内存开销小	|不稳定，易于陷入局部最优解|
|小批量梯度下降（MBGD）|	更稳定，计算速度快	|可能会受到批次大小的影响|
|动量梯度下降（Momentum）	|对梯度的变化进行平滑处理，能够加速收敛|	会有一定的超调现象|
|Nesterov加速梯度下降（NAG）|	在动量梯度下降基础上，进一步减少超调	|实现稍微复杂|
|自适应梯度下降（Adagrad）|	根据历史梯度信息调整学习率，适应性强	|学习率会不断减小，可能会导致较小的梯度被忽略|
|自适应矩估计梯度下降（Adam）|	结合Adagrad和动量梯度下降的优点，适应性强且收敛速度快|	计算复杂度高，可能会受到超参数的影响|

1. [详解梯度下降法（干货篇）](https://zhuanlan.zhihu.com/p/112416130)
2. [梯度下降法算法总结](https://blog.csdn.net/qq_35456045/article/details/104508217)
3. [梯度下降算法总结，至2020最新进展](https://zhuanlan.zhihu.com/p/294799487)

### 准确度(Accuracy)、精确度(Precision)、召回率(Recall)
- TP(True Positive): 被**正确**预测为有缺陷
- TN(True Negative): 被**正确**预测为非缺陷
- FP(False Positive): 被**错误**地预测为非缺陷的瓷砖实际上是有缺陷的
- TN(False Negative): 被**错误**地预测为有缺陷的瓷砖实际上是非缺陷的

		公式如图
	![](https://github.com/dhf97/image/raw/main/APR.png)

在不同的应用场景中，准确度、精度和召回率的重要性可能会有所不同。一般来说：

- **当在应用中需要将正例和负例区分开来的情况下，精度(Precision)是一个更为关键的指标**。例如，在医疗诊断中，我们需要保证诊断结果的准确性，而将健康的患者误诊为患有疾病，会给患者带来不必要的恐慌和治疗。因此，在这种情况下，精度比召回率更重要。
- **当需要尽可能识别出所有正例的情况下，召回率(Recall)是一个更为关键的指标。**例如，在病毒检测中，我们需要尽可能地识别出所有患有病毒的人，即使这意味着我们会将一些健康的人错误地诊断为患有病毒。在这种情况下，召回率比精度更重要。
- **当正例和负例都同等重要时，准确度(Accuracy)是一个更为关键的指标。**例如，在广告点击率预测中，我们需要预测哪些广告会被用户点击，同时避免将没有被点击的广告错误地预测为被点击。在这种情况下，准确度是最重要的指标。

# 六、结果：
## 对CBAM模块设置参数的性能评估
实验是通过改变通道模块的降维比率来进行的。
### 降维比率
指的是在注意力机制中，对于每个通道（channel）的特征图，通过一个降维操作将其压缩成一个标量（scalar）值，用于表示该通道的重要程度。该降维比率就是指将每个通道压缩成标量时所采用的维度大小，通常用一个称为“R”的参数来表示。例如，如果降维比率为8，则会将每个通道的特征图压缩成一个8维的向量。如果降维比率为16，则会将每个通道的特征图压缩成一个16维的向量，以此类推。降维比率越大，压缩后的向量维度就越小，因此每个通道所包含的信息量也就越少。
##### 结果：采用16的降维比率可以实现完美准确度
### CBAM模块数量和位置的影响
在**第2个卷积层(conv2)和第4个卷积层(conv4)后**分别集成两个CBAM模块可以产生良好的缺陷分类性能，其中精度为99.93%，精确度为100%，召回率为99.92%。此外，在CBAM模块通过conv1提取相关特征时，可以明显地分类缺陷，其中准确度为99.86%、精确度为99.92%、召回率为99.92%。

![](https://github.com/dhf97/image/raw/main/TEPN.png)

### 学习率的影响
当学习率过高 0.01 时，验证精度会降低。当学习率为 0.001 时，获得最佳性能。精度值达到99.93%。

### 分类类型的影响
使用了两个数据集来证明模型的有效性。数据集1是一个二元数据集，其中瓷砖被分为缺陷和非缺陷两类，而数据集2是一个多元数据集，其中非缺陷类别的图像被分为三个不同的分布，对应于我们的研究考虑的三种缺陷类型：角落破损，针孔和图案不连续。

##### 结果：在二元分类的情况下，模型更加有效地进行泛化，使准确度、精度和召回率分别提高了4.18%、4.09%和0.82%。
## 掩码的影响
通过基于分割和形态学运算以及HSV颜色模式的缺陷图像掩码的添加，我们实现了2.63%和2.44%的准确度和精度的性能提升。
## 跟其他模型的定量比较
在准确性方面优于其他四个模型。此外，与现有模型相比，文中的模型以显著较少的参数产生了良好的结果。
## 消融实验
|
|模型 |准确度（%） |精度（%） |召回率（%）|
|CNN_base |98.98| 99.84 |99.37|
|我们的模型 |99.93 |100 |99.92|
结果表明CBAM注意力模块使模型能够聚焦于瓷砖的相关特征以定位缺陷，从而提高模型的性能。

# 七、讨论：
模型在分类不同类型的缺陷方面表现更好。模型的二元分类缺陷分类准确度优于大多数先进模型，包括基于分割方法和带有机器学习算法的特征提取方法的模型[6-9]，以及基于深度学习的模型[11, 12]。这与所提出的预处理和分类步骤的效率和稳健性有关。一方面，增加的缺陷图像通过改善缺陷的视觉表示，通过添加掩模来表示。另一方面，整合CBAM注意力模块。此外，我们的模型通过减少参数数量降低计算成本。**缺点：即对于多类分类，缺陷检测率降低。**

多类分类是指将数据分成三个或三个以上类别的分类问题。在瓷砖缺陷检测中，多类分类问题是指要将瓷砖的不同缺陷类型分类，这比只需要将缺陷与非缺陷分类更具挑战性。



