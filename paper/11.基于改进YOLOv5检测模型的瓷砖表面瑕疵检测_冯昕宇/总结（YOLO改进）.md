本总结主要记录对传统YOLO算法的优化改进部分：



# YOLO原本结构：

输入端：预处理数据（进行包括平移颜色等的数据增强），锚框（预先设计的目标检测先验框），对图片进行缩放

Backbone：特征提取：

Neck ：对 Backbone 各层提取的特征信息进行融合

 Prediction ：预测输出层

![image-20230405103952817](https://user-images.githubusercontent.com/103879136/230783214-66f34d5c-dad5-4b3d-80a4-f0e8ede381d1.png)





# 优化操作：

## SE注意力模块

在神经网络中添加注意力机制，进而使得网络能够从图像中筛选出重要的信息：

本论文提到了4个轻量级（指对模型速度影响不大）注意力模块，以下是对比分析后的结果：


![image-20230405153109858](https://user-images.githubusercontent.com/103879136/230783350-2ea08566-e3f3-42d2-b218-656b2f96f230.png)

通过以上对比分析，添加 SE 注意力模块后，模型在瓷砖瑕疵数据集上的平均检测精度 mAP 提升最大，并且 SE 模块是一个轻量化的模块，加入到 YOLOv5 中几乎 不会影响模型的大小和检测速度，所以将添加 SE 注意力模块的 YOLOv5 作为后续优 化的基础。

以下简要介绍SE注意力模块：

SE （Squeeze-and-Excitation）注意力模块分为两个过程：

Squeeze：将一个通道上的 空间信息压缩为一个全局特征，具体做法是通过一个全局平均池化（global average  pooling）将每个通道上的二维特征编码为一个实数，其表达式为：

![image-20230405153257867](https://user-images.githubusercontent.com/103879136/230783361-a1e3cde8-ec23-4ea2-900e-abaab561adac.png)

Excitation：是为每个压缩后的特征图的每个通道动态生成不同的权重值，具体做法是通过使用两个全连接层（FC）组成一个 Bottleneck 结构去构建通道 间的相关性，输出每个通道的权重值。其表达式为：

![image-20230405153415606](https://user-images.githubusercontent.com/103879136/230783367-64be6ba2-fca9-4b80-b919-ca00205a6d88.png)

全过程流程如下图所示：

![image-20230405142708825](https://user-images.githubusercontent.com/103879136/230783392-49c4d32c-1c8d-435a-9d1d-99fc3590d994.png)

与此同时，SE注意力机制还能嵌入ResNet中提高效率：

![image-20230405154843150](https://user-images.githubusercontent.com/103879136/230783407-46c8cf0e-6f9c-402f-a462-767c2c8d4902.png)
## 双向特征融合BiFPN

![image-20230405155941885](https://user-images.githubusercontent.com/103879136/230783414-59bfc55d-e6ab-4fe1-9dc5-f793127e2947.png)

通过实验可以发现对 YOLOv5s 的特征融合结构进行改进后，在瓷砖瑕疵检测数 据集上的平均检测精度提升了 1.26 个百分点，并且对小目标的检测精度也有所提升， 再添加 SE 注意力模块后，又提升了 0.9 个百分点，说明改进特征融合结构对提升 YOLOv5s 检测模型再瓷砖瑕疵数据集上的性能是有效的。

以下对BiFPN进行介绍：

在YOLOv5 中的特征融合结构为 PAN 结构，相当于在自顶向下的 FPN 结构中添加了自底向上的结构：

![image-20230405161722738](https://user-images.githubusercontent.com/103879136/230783419-deb21a42-f1e4-407e-abee-3351b429b73d.png)

但在BiFPN中进行特征融合前会对每个输入特征图增加一个额外的权值，让网络学习 不同输入特征图的重要程度，为了尽量减少特征融合的时间成本，采用如下图所示的快速归一化的融合策略：

![image-20230405161937609](https://user-images.githubusercontent.com/103879136/230783424-cc21fd49-4bf0-4f0f-84d7-618e59b13c95.png)

在其结构中，BiFPN不仅删除了只有一个输入的节点（如P7和P3），还引入了跨通道的跳跃连接（虚线箭头）其结构如下图所示

![image-20230405162229503](https://user-images.githubusercontent.com/103879136/230783431-002aa1b1-7e54-4fd3-909f-4d0a42e8954a.png)

## 结构重参化RepVGG：改进卷积块

在引入残差结构的基础上，采用 RepVGG[54]中的思想，对残差结构进行结构重参数化，其结构如图所示：

![image-20230405163612543](https://user-images.githubusercontent.com/103879136/230783436-faa01651-13b6-4f72-adb4-aea2a1218d19.png)

对比ResNet（每隔两层或三层增加一个分支结构），RepVGG 在每一个 3×3 的卷积层都平行增加一个 1×1 的卷积层分支和一个恒等映射分支，构成一个 RepVGG Block 这样可以提升训练的性能。（下面还介绍了卷积的可加性）

最后实验结果如图所示：

![image-20230405163728652](https://user-images.githubusercontent.com/103879136/230783442-e57be747-255f-49d2-8452-0cc925bea9e5.png)

将这种策略和前面所述策略结合后，发现准确率能够提高3.47 个百分点

# 基于YOLOv5本身的轻量化改进

由于这些轻量化改进会有牺牲检测精度换取计算量的情况，所以在项目初级阶段先不予考虑

