## 论文《基于改进 YOLOv3 的纹理瓷砖缺陷检测》背景、研究现状摘抄，我们可以选取需要的论文，rewrite
我国是瓷砖生产大国，国内的墙地砖生产企业在原料加工方面已实现了机械化和自动化，但在缺陷检测方面大多仍停留在人工检测的水平 ，检测效率稳定性都难以满足自动化生产要求。在瓷砖的生产流程中，从原料、配方、球磨、制粉、成形、烧制到磨边倒角等，各个环节均会产生不同的缺陷，其中主要包括孔洞、划痕、杂斑等类型表面缺陷[1]。    
![image](https://user-images.githubusercontent.com/103879136/231338036-9683cebc-53f8-4858-bfff-70c70b859963.png)
[图片来源](https://www.hanghangcha.com/cms/detail/11849.html)(用到时剪掉上下部分)  
如Ahamad等[2]先对瓷砖进行亮度增强和中值滤波的预处理，然后分割瓷砖图像并进行特征提取，最后利用图像直方图检测瓷砖缺陷，实现了对瓷砖缺陷的分类。Hanzaei等[3]用旋转不变测量局部方差算子特征提取缺陷特征，经过形态学增强后利用支持向量机进行缺陷分类。艾矫燕[4]提出一种基于马尔可夫随机场纹理分析的墙地砖色彩缺陷检测方法，而对于孔穴类型缺陷，则利用基于灰度直方图的阈值分割方法进行检测。脱羚[5]对瓷砖尺寸及表面缺陷开展研究，提出了一种基于多特征融合自适应模糊系统的缺陷检测方法，对瓷砖的黑斑、开裂、釉裂、缺釉、麻面5种缺陷进行识别。李小磊[6]等提出一种基于滑动滤波和形态学处理的陶瓷瓦表面裂纹缺陷检测方法，该方法可以去除光照不均和纹理对裂纹检测的影响。李强[7]等 基于主成分分析（PCA）法对瓷砖图像进行重构，再与原图进行差分，然后利用形态学和二值化方法提取缺陷区域，准确率达96％。上述方法过于依赖人工设计的特征提取算子，在复杂工业环境中存在鲁棒性不足、硬件调试复杂、算法参数调试困难等缺点[8]。另一类则是随着深度卷积神经网络发展而出现的目标检测算法，目前深度学习在工业检测领域已经受到了广泛的关注。He[9]等提出的基于Faster-RCNN[10]的带钢表 面缺陷检测网络的各类别平均精确度的平均值（mAP）在缺陷检测数据集NEU—DET 上达82.3％。李东洁等[11]利用融合特征金字塔网络（FPN）的Faster-RCNN对马克杯缺陷进行识别， 相比原Faster—RCNN算法，准确率提高了2.5％。周君等[12]提出基于改进YOLOv3的织物缺陷实时检测方法，通过对YOLOv3模型进行剪枝压缩，在不改变检测准确率的前提下提高了网络的预测速度。刘露露等[13]提出了基于 YOLOv3 的布匹瑕疵检测算法，检测准确率可达 89. 66%。   

[1]李泽辉,陈新度,黄佳生等.基于改进YOLOv3的纹理瓷砖缺陷检测[J].激光与光电子学进展,2022,59(10):294-302.   
[1]李军华, 权小霞, 汪宇玲 . 多特征融合的瓷砖表面缺陷检测算法研究[J]. 计算机工程与应用, 2020, 56(15): 191-198.   
Li J H, Quan X X, Wang Y L. Research on defect detection algorithm of ceramic tile surface with multi-feature fusion[J]. Computer Engineering and Applications, 2020, 56(15): 191-198.   
[2]Ahamad N S, Rao J B. Analysis and detection of surface defects in ceramic tile using image processing techniques[M]∥Satapathy S C, Rao N B, Kumar S S, et al. Microelectronics, electromagnetics and telecommunications. Lecture notes in electrical engineering. New Delhi: Springer, 2016, 372: 575-582.   
[3] Hanzaei S H, Afshar A, Barazandeh F. Automatic detection and classification of the ceramic tiles’ surface defects[J]. Pattern Recognition, 2017, 66: 174-189.   
[4]艾矫燕. 基于计算机视觉的墙地砖颜色分类和缺陷检测研究[D].华南理工大学,2003.   
[5]脱羚. 基于多特征信息融合的陶瓷砖表面质量在线检测方法研究[D].陕西科技大学,2019.   
[6]李小磊,曾曙光,郑胜等.基于滑动滤波和自动区域生长的陶瓷瓦表面裂纹检测[J].激光与光电子学进展,2019,56(21):49-55.   
[7]李强,曾曙光,郑胜等.基于机器视觉的陶瓷瓦表面裂纹检测方法[J].激光与光电子学进展,2020,57(08):51-57.   
[8]陶显,侯伟,徐德.基于深度学习的表面缺陷检测方法综述[J].自动化学报,2021,47(05):1017-1034.DOI:10.16383/j.aas.c190811.   
[11]李东洁,李若昊.基于改进Faster RCNN的马克杯缺陷检测方法[J].激光与光电子学进展,2020,57(04):353-360.   
[12]周君,景军锋,张缓缓等.基于S-YOLOV3模型的织物实时缺陷检测算法[J].激光与光电子学进展,2020,57(16):55-63.   
[13]刘露露,李波,何征等.基于FS-YOLOv3及多尺度特征融合的棉布瑕疵检测[J].中南民族大学学报(自然科学版),2021,40(01):95-101.   
***
## 摘自：机器视觉表面缺陷检测综述
汤勃,孔建益,伍世虔.机器视觉表面缺陷检测综述[J].中国图象图形学报,2017,22(12):1640-1663.   
中国是一个制造大国，每天都要生产大量的工业产品。用户和生产企业对产品质量的要求越来越高，除要求满足使用性能外，还要有良好的外观，即良好的表面质量。但是，在制造产品的过程中，表面缺陷的产生往往是不可避免的。不同产品的表面缺陷有着不同的定义和类型，一般而言表面缺陷是产品表面局部物理或化学性质不均匀的区域，如金属表面的划痕、斑点、孔洞，纸张表面的色差、压痕，玻璃等非金属表面的夹杂、破损、污点，等等。表面缺陷不仅影响产品的美观和舒适度，而且一般也会对其使用性能带来不良影响，所以生产企业对产品的表面缺陷检测非常重视，以便及时发现，从而有效控制产品质量，还可以根据检测结果分析生产工艺中存在的某些问题，从而杜绝或减少缺陷品的产生，同时防止潜在的贸易纠份，维护企业荣誉。
***
# 论文[《基于改进YOLOv5检测模型的瓷砖表面瑕疵检测》](https://kns.cnki.net/kcms2/article/abstract?v=Emm5A0E9Hv1istwDMhFD4HEnu927KUVPXPmPmLZXqn2o_DsObp-bl-25e_gXCHSEHmBUoWK_1ldWAjbUwd1wiRqyaWu6jR1UtZBxaWhN9cU1eV-Pj-FB4DX0IFsRRG_HnFwhPANLoJHWJYPM5MBZITdlgJtgdMfz&uniplatform=NZKPT)研究意义，研究现状摘抄
## 研究意义
目前，我国大多数瓷砖生产厂商主要是通过人力进行瑕疵检测。由于瓷砖瑕疵大多尺寸较小且砖面光滑反光，需要在一定光照以及合适的角度才能清楚地发现瑕疵。面对快速且大批量的生产需求，人力检测的缺点主要体现在：（1）人力检测精度较低、不同人检测标准不一致、漏检错检率较高；（2）人力检测效率偏低，与快速大批量的流水线作业不匹配；（3）人工成本越来越高且人工容易出现疲劳状态，反而加大检测成本。随着产业自动化水平的不断提高，人力检测瓷砖瑕疵已经不能满足大规模生产线的要求，先进的自动检测技术在不断提高。近几年，计算机视觉、深度学习等理论迅速发展，尤其是基于深度卷积神经网络的检测方法，其可以对图像信息进行特征提取，并实现分类及检测。机算机视觉技术[3]已经在钢材、铝材、PCB 板等缺陷检测领域有了很多的应用，不仅具有安全可靠、使用灵活、可以长时间使用等优点，未来还有可能完全取代人工，实现全面的自动化高质量生产。但目前，我国瓷砖瑕疵自动化检测水平并不高，其中一个重要的原因是缺乏较高检测精度和速度的检测算法。所以，以瓷砖瑕疵检测为研究对象，提出适用于瓷砖瑕疵检测的算法，将其应用于瓷砖瑕疵检测设备中，对提高瓷砖生产质量有很重要的意义。
## 研究现状
(感觉除了分小点罗列技术，还需要列出技术本身的局限性，才能凸显我们方法的优势)  
基于传统机器学习的目标检测模型属于浅层模型，其特点是需要根据人工经验抽取样本特征，而模型主要负责分类或预测。图 1-2 为传统机器学习算法的目标检测流程，一般使用一个滑动窗口在图片上进行滑动，利用方向梯度直方图[5]（HOG）以及尺度不变特征变换[6]（SIFT）等人工选取的特征处理算子对滑动窗口区域进行特征提取，然后使用支持向量机[7]（SVM）等分类算法对提取到的特征进行分类，以此达到目标检测的目的。  
![image](https://user-images.githubusercontent.com/103879136/231125997-55601eb4-5221-458e-a3d6-1442cc04cf8d.png)
### 基于传统机器学习的检测方法
对于使用机器学习算法进行缺陷检测，国内外的学者都进行了一些研究。邹庆胜[8]等人提出一种基于图像处理的瓷砖瑕疵检测系统，通过对图像的颜色、尺寸偏差等参数判断瓷砖是否存在缺陷，适用于目标与背景差异较大的目标定位。周善旻[9]通过建立了颜色纹理特征库，然后利用 Filter 和 Wrapper 结合的算法对瓷砖表面缺陷进行了分级检测，不同于表面缺陷检测，表面分级应用通常采用彩色图像处理，同时考虑纹理和颜色特征。Saeed[10]等人提出一种具有高精度的自动图像处理系统，利用 RIMLV算子进行缺陷边缘检测，并配合使用闭合形态算子对检测区域进行填充和平滑处理。段春梅[11]等人首先采取双边滤波器对原始图像进行降噪处理，然后利用 Canny 算子对图像提取边缘特征，根据阈值对图像进行分割，利用圆形度对缺陷特征进行描述。谢波[12]等人提出基于图像梯度方差和信息熵结合的 BHPF 检测算法，对瓷砖瑕疵进行特征提取。张军[13]提出了基于阈值分割地检测算法以及基于形态学和小波变化的瓷砖缺陷检测方法，并在硬件设备上进行了测试。高倩倩[14]提出基于融合数学形态学闭运算的局部方差选择不变测度的边缘检测算法。脱羚[15]提出基于 TMOAA 算法的瓷砖表面质量的在线检测系统，利用 LED 光源以及 CCD 相机搭建了完整的瓷砖瑕疵在线检测平台。  
上述基于机器学习的瓷砖瑕疵检测研究，主要是根据实践经验以及具体图片特征，人为选取特征提取算子对图像特征进行提取，最后使用分类器对瑕疵进行分类。特征选取的好坏决定了模型的检测性能，在某些比较容易确定特征的场景，比如纯色瓷砖，且瑕疵本身与背景对比度比较大时，该类方法可以取得不错的检测效果，但是鲁棒性较差，面对复杂多变的生产环境检测效果会大幅降低。总体而言，基于机器学习的检测算法依赖人工设计特征提取算子，并且人工设计的特征提取算子往往只能对图像的浅层特征进行提取，泛化能力不强，对于复杂的检测任务检测效果一般。 

### 基于深度学习的检测方法
得益于算力设备的不断升级，深度学习技术在目标检测领域取得了巨大进展，在缺陷检测方面也得到广泛应用。基于深度学习的目标检测算法不同于传统机器学习采用特征算子进行特征提取，深度学习算法一般采用深度卷积神经网络进行特征提取，往往更深的网络可以提取到更复杂的图像信息，所以基于深度学习的检测算法可以应用于更复杂的检测任务。最开始深度学习主要运用于分类任务，诞生了 AlexNet[16]、VGG[17]、GoogleNet[18]、ResNet[19]等图像分类网络，基于深度学习的图像分类算法可以对图像特征进行更深层次的提取，在多项图像分类任务大赛中都远远超过了传统的机器学习算法，这些图像分类算法往往作为目标检测算法的基础，起到特征提取的作用。2014 年，Ross Girshick[20]等人将已经在分类任务中取得很好成绩的卷积神经网络应用到目标检测任务中，提出了基于深度学习的目标检测的开山之作——RCNN，该方法的中文直译为“具有 CNN 特征的区域”（Regions with CNN features），该方法一经问世就刷新了记录，在 PASCAL VOC 数据集上，将目标检测的平均检测精度提升到 53.3%，较之前最好的检测结果提升超过 30%。后来基于深度学习的目标检测算法得到飞速发展，涌现出许多优秀的目标检测算法，可以将基于深度学习的目标检测算法根据是否生成候选框分为两阶段算法和单阶段算法。图 1-3 和 1-4 分别为两阶段和单阶段检测算法的示意图，两阶段比较常用的有 RCNN、Fast-RCNN[21]以及 FasterRCNN[22]等算法，一阶段比较常用的有 SSD[23] (Single Shot MultiBox Detector) 以及YOLO[24-27] (You Look Only Once) 系列算法。通常两阶段算法有更高的检测精度，单阶段算法相对来说检测精度可能略低但检测速度一般较高。  
![image](https://user-images.githubusercontent.com/103879136/231127055-5adc1987-8b76-4544-a7bc-7747c2d53139.png)
深度学习技术作为人工智能的重大分支，近年来在工业检测领域中取得了巨大的进展，在缺陷检测中表现出广阔的应用前景。陈长虹[28]等人利用布尔神经网络对瓷砖瑕疵进行特征提取。张涛川[29]等人提出一种基于双流卷积神经网络的瓷砖缺陷检测算法，使用最大值融合策略对模型的特征进行融合。Li[30]等人利用 YOLOv3 目标检测算法实现了对 6 种瓷砖缺陷进行检测。Zhang[31]等人提出一种基于 YOLOv3 网络体系结构的智能缺陷检测方法，提出 K-medoids 聚类算法策略，提高了对小缺陷的检测性能。Lian[32]等人使用知识蒸馏、模型压缩等方法对 YOLOv4 检测算法进行改进，并加入更细致的感受野预测尺度，提高了对小目标检测的性能。  
上述主要介绍了研究人员在缺陷检测中，使用深度学习算法进行检测的案例，可以说明使用深度学习进行缺陷检测的有效性。但由于在工业界没有标准的瓷砖瑕疵检测数据集，其中进行瓷砖瑕疵检测的研究人员大多使用自己收集的数据集进行训练和测试，这些数据集往往质量参差不齐，不具有可比性。目前，在瓷砖瑕疵检测任务中还存在缺乏数据集，检测算法有待优化等问题。


## 参考文献
[3]汤勃, 孔建益, 伍世虔. 机器视觉表面缺陷检测综述[J]. 中国图象图形学报, 2017, 22(12): 1640-1663.   
[5] Dalal N, Triggs B. Histograms of oriented gradients for human detection[C].//IEEE Computer Society Conference on Computer Vision and Pattern Recognition(CVPR2005). IEEE, 2005, 1:886-893.   
[6] Lowe D G. Distinctive image features from scale-invariant keypoints[J]. International Journal of Computer Vision, 2004, 60(2): 91-110.   
[7] Burges C. A Tutorial on Support Vector Machines for Pattern Recognition[J]. Data Mining and Knowledge Discovery, 1998, 2(2): 121-167.  
[8] 邹庆胜, 汪仁煌, 明俊峰. 基于机器视觉的瓷砖多参数分类系统的设计[J]. 广东工业大学学报, 2010, 27(04): 46-49.   
[9] 周善旻. 产品复杂表面质量的工业视觉检测方法和应用研究[D]. 宁波大学, 2017.   
[10] Hanzaei S H, Afshar A, Barazandeh F. Automatic detection and classification of the ceramic tiles’surface defects[J]. Pattern Recognition, 2017. 66(03): 174-189.  
[11] 段春梅, 张涛川. 基于机器视觉的瓷砖素坯表面缺陷无损检测算法研究[J]. 智能计算机与应用, 2017, 7(03): 37-40.   
[12] 谢波, 张平. 基于机器视觉的墙地砖表面缺陷检测系统研究[J]. 机械工程与自动化, 2017(05): 130-132.   
[13] 张军. 基于数字图像处理的瓷砖表面缺陷检测研究[D]. 山东理工大学, 2018.   
[14] 高倩倩. 瓷砖表面质量视觉检测技术研究[D]. 山东理工大学, 2018.   
[15] 脱羚. 基于多特征信息融合的陶瓷砖表面质量在线检测方法研究[D]. 陕西科技大学, 2019.   
[16] Krizhevsky A, Sutskever I, Hinton G E. ImageNet Classification with deep convolutional neural networks[J]. Advances in Neural Information Processing Systems, 2012, 25(2): 1097-1105.   
[17] Mateen M, Wen J, Song S, et al. Fundus image classification using VGG-19 architecture with PCA and SVD[J]. Symmetry, 2018, 11(1): 1.   
[18] Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 1-9.   
[19] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778.   
[20] Girshick R, Donahue J, Darrell T, et al. Rich feature hierarchies for accurate object detection and semantic segmentation[C]//Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2014: 580-587.   
[21] Girshick R. Fast R-CNN[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1440- 1448.   
[22] Ren S, He K, Girshick R, et al. Faster R-CNN: Towards real-time object detection with region proposal networks[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2017, 39(6): 1137-1149. 
[23] Liu W, Anguelov D, Erhan D, et al. SSD: Single shot multibox detector[C]//European Conference on Computer Vision. Cham: Springer, 2016: 21-37.   
[24] Redmon J, Divala S, Girshick R, et al. You only look once: unified, real-time object detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 779-788.   
[25] Redmon J, Farhadi A. YOLO9000: better, faster, stronger[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017: 6517-6525. 
[26] Redmon J, Farhadi A. YOLOv3: an incremental improvement[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018: 89-95.   
[27] Bochkovskiy A, Eang C Y, Liao H-Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection [EB/OL]. (2020-04-23)[2020-11-14].https://arxiv.org/pdf/2004.10934.pdf.   
[28] 陈长虹. 瓷砖表面缺陷检测中布尔神经网络方法的运用[J]. 中国高新技术企业, 2013(30): 31-32.   
[29] 张涛川, 段春梅. 基于深度学习的瓷砖在线快速无损检测系统开发[J]. 机械工程与自动化, 2020(06): 129-130+133.   
[30] Li G, Liu X, Tao B, et al. Research on ceramic tile defect detection based on YOLOv3[J]. International Journal of Wireless and Mobile Computing, 2021, 21(2): 128-133.   
[31] Zhang Z, Zhang Y, Wen Y, et al. Intelligent Defect Detection Method for Additive Manufactured Lattice Structures Based on a Modified YOLOv3 Model[J]. Journal of Nondestructive Evaluation, 2021, 41(1): 1-14.   
[32] Lian J, He J, Niu Y, et al. Fast and accurate detection of surface defect based on improved YOLOv4[J]. Assembly Automation, 2022, 42(1): 134-146. 
***
可以找一些中国瓷砖市场发展的图
