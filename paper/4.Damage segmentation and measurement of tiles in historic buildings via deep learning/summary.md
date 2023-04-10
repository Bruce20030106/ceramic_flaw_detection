# 故宫瓦片图像的自动裁剪和检测
**总结：论文提出了一种基于深度学习技术的两级策略，实现了对历史建筑屋顶上的琉璃瓦的自动损伤检测、分割和测量的方法。首次使用两级策略基于深度学习自动检测、分割和测量历史建筑上的大规模表面损伤。**
1. 第一级利用Faster R-CNN方法从屋顶图像中自动检测和裁剪两种类型的琉璃瓦，并将裁剪后的图像用于训练Mask R-CNN模型。
2. 第二级利用Mask R-CNN方法对裁剪后的琉璃瓦图像进行损伤分割和测量，并基于Mask R-CNN的预测结果定量评估损伤的形态特征，如损伤拓扑、面积和比例。
***
## Data Preparation
### 训练Faster R-CNN
第一部分是基于收集的屋顶图像准备数据集，用于训练Faster R-CNN模型。作者使用iPhone 6拍摄了400张屋顶图像，并使用LabelImg软件标注了两种类型的琉璃瓦的类别和边界框坐标。然后将80张图像作为测试集，剩余320张图像随机分为训练集和验证集。   
![](https://github.com/OctoberEnd/verbose-invention/blob/main/pic/png.png?raw=true)    
### 训练Mask R-CNN模型
第二部分是基于第一部分裁剪得到的琉璃瓦图像准备数据集，用于训练Mask R-CNN模型。作者从裁剪后的图像中选择了500张琉璃瓦图像，并根据损伤程度将其分为四类：简单损伤、中等损伤、严重损伤和极度损伤。然后使用Photoshop软件标注了每张图像上的损伤区域，并计算了损伤面积和比例。
![](https://github.com/OctoberEnd/verbose-invention/blob/main/pic/pic4.png?raw=true)   
论文中的binary mask format是用于实例分割的一种方法，它可以为每个感兴趣区域（ROI）生成一个二值掩码，表示该区域内的像素是否属于目标对象。
为了方便网络训练，作者将样本转换为binary mask format，即将每张图像上的损伤区域用白色像素表示，其余部分用黑色像素表示。这样就可以用Mask R-CNN模型对每个ROI进行像素级别的损伤分割。
***


## Model Training
### Faster R-CNN是如何训练的
Faster R-CNN包含两个核心组件——区域提议网络（RPN）和Fast R-CNN。RPN和Fast R-CNN网络都使用相同的CNN从原始图像中提取特征。   
   
论文中的Faster R-CNN模型是使用ZF-Net作为CNN来提取特征，然后使用RPN来生成候选区域，最后使用全连接层来输出瓦片的类别和边界框。
![]()
全连接层是指神经网络中的一种层，它由权重、偏置和神经元组成，用于将两个不同层的神经元完全连接起来。全连接层通常放在输出层之前，构成CNN架构的最后几层。全连接层的作用是利用卷积过程中提取的特征，来预测图像的类别或其他信息。在论文中，全连接层是用来输出瓦片的类别和边界框的。  
   
在RPN中，使用了一种叫做**锚点anchor**的概念来生成候选区域。锚点实际上就是一些固定大小和宽高比的矩形框，它们被放置在输入图像的每个位置上，并作为可能包含目标的候选区域。这些锚点的大小和宽高比通常是预先定义好的，例如在Faster R-CNN中，使用了三种不同的尺度和三种不同的宽高比，共计九种锚点。在RPN中，每个锚点都会输出两个分数，表示它是前景（即包含目标）和背景（即不包含目标）的概率。通过对这些锚点进行非极大值抑制（NMS）操作，就可以得到最终的候选区域。   
***
### Mask R-CNN是如何训练的
使用ResNet101和FPN作为CNN来提取特征，然后使用RPN来生成候选区域，最后使用ROI align和全连接层来输出瓦片的类别、边界框和分割掩码。  
![](https://github.com/OctoberEnd/verbose-invention/blob/main/pic/pic2.png?raw=true)   
#### 模型初始化Model initialization
在训练Mask R-CNN模型之前，使用预训练的ResNet101模型来初始化卷积层的参数。这样可以节省训练时间，也可以提高模型的泛化能力。   
#### 具体步骤
- 使用预训练的ResNet101模型来初始化Mask R-CNN的卷积层参数，以节省训练时间和提高泛化能力。
- 使用每个GPU一张图片的小批量数据来训练模型，每个图片有N个采样的感兴趣区域（ROI），其中N为64或512。
- 使用多任务损失函数来优化模型，包括分类损失、边界框损失和掩码损失。为了训练这个模型，定义一个合适的损失函数，来衡量预测结果和真实标签之间的差异
***
- **以下12345点为具体函数说明**。
![](https://github.com/OctoberEnd/verbose-invention/blob/main/pic/pic1.png?raw=true)
1. pi————>表示第i个锚点（Faster R-CNN生成）框选住目标的概率；   
   pi*————>是由loU评估的目标分类。换而言之，当值为0，表示是背景；当值是1，表示是目标   
   t——>表示中心到两个方向上的缩放,"a"表示锚点的位置，“星号*”表示最近目标的实际位置
   ![](https://github.com/OctoberEnd/verbose-invention/blob/main/pic/loss.png?raw=true)
2. loU是一种测量在特定数据集中检测相应物体准确度的一个标准,用于测量真实和预测之间的相关度，相关度越高，该值越高。
3. ![](https://github.com/OctoberEnd/verbose-invention/blob/main/pic/loU.png?raw=true)    
4. ![](https://github.com/OctoberEnd/verbose-invention/blob/main/pic/loU1.png?raw=true)   
5. Lmask——>计算ROI上的所有像素的平均交叉熵，（loU大于0.5的ROI部分）
![](https://github.com/OctoberEnd/verbose-invention/blob/main/pic/mask.png?raw=true)
***
- 使用ROI对齐层来替代ROI池化层，以避免ROI边界或区域的量化误差，提高预测掩码的精度。
- 使用学习率为0.001，动量为0.9，权重衰减为0.0001的随机梯度下降法来更新参数，每10k步将学习率减少10倍。
- 先只训练网络头部的随机初始化层，然后再微调所有层，总共进行40个周期的训练。
   
**ROI**是Region of Interest的缩写，意思是**感兴趣区域**。在计算机视觉中，ROI通常指的是图像中包含有用信息的一部分，例如目标对象的边界框或分割掩码。在论文中，ROI代表的是瓦片图像中的损伤区域，也就是Mask R-CNN要分割和测量的对象。   
![](https://github.com/OctoberEnd/verbose-invention/blob/main/pic/pic3.png?raw=true)   
  
**掩码分支**是指Mask R-CNN模型中的一个分支，它用于对每个候选区域进行像素级的分割，输出K×m×m的二值掩码，其中K是类别数，m是掩码的大小。掩码分支是一个全卷积网络，它使用ROI align层来提取每个候选区域的特征图，然后使用几个卷积层和反卷积层来生成掩码。掩码分支只对正样本（与真实边界框的IoU大于0.5的候选区域）有效，且只输出与预测类别标签一致的Ki类别的掩码。最终，所有预测的掩码可以用来表示损伤区域的形状和位置。   
   
掩码分支中的**掩码**是指一个二值矩阵，它表示一个候选区域中的哪些像素属于某个类别，哪些像素不属于。掩码中的1表示属于该类别的像素，0表示不属于该类别的像素。例如，对于瓦片的损伤分割，掩码中的1表示损伤区域的像素，0表示正常区域的像素。掩码可以用来可视化分割结果，也可以用来计算损伤的面积和比例。
