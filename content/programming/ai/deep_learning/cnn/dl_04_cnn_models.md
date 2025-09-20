---
title: "深度学习04-CNN经典模型"
date: 2025-09-18T16:55:17+08:00
# bookComments: false
# bookSearchExclude: false
---

# 简介
卷积神经网络（CNN）是深度学习中非常重要的一种网络结构，它可以处理图像、文本、语音等各种类型的数据。以下是CNN的前4个经典模型
1. LeNet-5

LeNet-5是由Yann LeCun等人于1998年提出的，是第一个成功应用于手写数字识别的卷积神经网络。它由7层神经网络组成，包括2层卷积层、2层池化层和3层全连接层。其中，卷积层提取图像特征，池化层降低特征图的维度，全连接层将特征映射到对应的类别上。

LeNet-5的主要特点是使用Sigmoid激活函数、平均池化和卷积层后没有使用零填充。它在手写数字识别、人脸识别等领域都有着广泛的应用。

2. AlexNet

AlexNet是由Alex Krizhevsky等人于2012年提出的，是第一个在大规模图像识别任务中取得显著成果的卷积神经网络。它由5层卷积层、3层全连接层和1层Softmax输出层组成，其中使用了ReLU激活函数、最大池化和Dropout技术。

AlexNet的主要特点是使用了GPU加速训练、数据增强和随机化Dropout等技术，使得模型的泛化能力和鲁棒性得到了大幅提升。它在ImageNet大规模图像识别比赛中取得了远超其他模型的优异成绩。

3. VGGNet

VGGNet是由Karen Simonyan和Andrew Zisserman于2014年提出的，它是一个非常深的卷积神经网络，有16层或19层。VGGNet的每个卷积层都使用了3x3的卷积核和ReLU激活函数，使得它的网络结构非常清晰、易于理解。

VGGNet的主要特点是使用了更深的网络结构、小卷积核和少量的参数，使得模型的特征提取能力得到了进一步提升。它在ImageNet比赛中也获得了非常好的成绩。

4. GoogLeNet

GoogLeNet是由Google团队于2014年提出的，它是一个非常深的卷积神经网络，有22层。它使用了一种称为Inception模块的结构，可以在保持网络深度的同时减少参数量。

GoogLeNet的主要特点是使用了Inception模块、1x1卷积核和全局平均池化等技术，使得模型的计算复杂度得到了大幅降低。它在ImageNet比赛中获得了非常好的成绩，并且被广泛应用于其他领域。
# CNN回顾
回顾一下 CNN 的几个特点：局部感知、参数共享、池化。
## 局部感知
人类对外界的认知一般是从局部到全局、从片面到全面，类似的，在机器识别图像时也没有必要把整张图像按像素全部都连接到神经网络中，在图像中也是局部周边的像素联系比较紧密，而距离较远的像素则相关性较弱，因此可以采用局部连接的模式（将图像分块连接，这样能大大减少模型的参数），如下图所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/ded35fd1891c4ae5860ca339ab2e9194.png)
## 参数（权值）共享
每张自然图像（人物、山水、建筑等）都有其固有特性，也就是说，图像其中一部分的统计特性与其它部分是接近的。这也意味着这一部分学习的特征也能用在另一部分上，能使用同样的学习特征。因此，在局部连接中隐藏层的每一个神经元连接的局部图像的权值参数（例如 5×5），将这些权值参数共享给其它剩下的神经元使用，那么此时不管隐藏层有多少个神经元，需要训练的参数就是这个局部图像的权限参数（例如 5×5），也就是卷积核的大小，这样大大减少了训练参数。如下图
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/5719b5393d7465c85c55dbfde33a9a52.png)
>卷积核的权值是指每个卷积核中的参数，用于对输入数据进行卷积操作时，对每个位置的像素进行加权求和。在卷积神经网络中，同一层中的所有卷积核的权值是共享的，这意味着每个卷积核在不同位置上的权值是相同的。共享权值可以减少模型中需要学习的参数数量，从而降低了模型的复杂度，同时可以提高模型的泛化能力，因为共享权值可以使模型更加稳定，避免过度拟合。共享权值的实现方式是通过使用相同的卷积核对输入数据进行卷积操作。
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/256075656a5495f06f01fca1556e8b15.png)

## 池化
随着模型网络不断加深，卷积核越来越多，要训练的参数还是很多，而且直接拿卷积核提取的特征直接训练也容易出现过拟合的现象。回想一下，之所以对图像使用卷积提取特征是因为图像具有一种 “静态性” 的属性，因此，一个很自然的想法就是对不同位置区域提取出有代表性的特征（进行聚合统计，例如最大值、平均值等），这种聚合的操作就叫做池化，池化的过程通常也被称为特征映射的过程（特征降维），如下图：


# LeNet-5
概述
LeNet5 诞生于 1994 年，是最早的卷积神经网络之一， 由 Yann LeCun 完成，推动了深度学习领域的发展。在那时候，没有 GPU 帮助训练模型，甚至 CPU 的速度也很慢，因此，LeNet5 通过巧妙的设计，利用卷积、参数共享、池化等操作提取特征，避免了大量的计算成本，最后再使用全连接神经网络进行分类识别，这个网络也是最近大量神经网络架构的起点，给这个领域带来了许多灵感。
LeNet5 的网络结构示意图如下所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/885ae480f4d134a7c635837766ad33f6.png)
LeNet5 由 7 层 CNN（不包含输入层）组成，上图中输入的原始图像大小是 32×32 像素，卷积层用 Ci 表示，子采样层（pooling，池化）用 Si 表示，全连接层用 Fi 表示。下面逐层介绍其作用和示意图上方的数字含义。

## C1 层（卷积层）：6@28×28
该层使用了 6 个卷积核，每个卷积核的大小为 5×5，这样就得到了 6 个 feature map（特征图）。
（1）特征图大小
每个卷积核（5×5）与原始的输入图像（32×32）进行卷积，这样得到的 feature map（特征图）大小为（32-5+1）×（32-5+1）= 28×28
卷积过程如下图所示（下图是4*4只是用于演示）：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/40ca415abf37addb1558b70f12e2fb83.png)
卷积核与输入图像按卷积核大小逐个区域进行匹配计算，匹配后原始输入图像的尺寸将变小，因为边缘部分卷积核无法越出界，只能匹配一次，如上图，匹配计算后的尺寸变为 Cr×Cc=（Ir-Kr+1）×（Ic-Kc+1），其中 Cr、Cc，Ir、Ic，Kr、Kc 分别表示卷积后结果图像、输入图像、卷积核的行列大小。
其中Cr表示结果行row，Cc表示结果列column
（2）参数个数
由于参数（权值）共享的原因，对于同个卷积核每个神经元均使用相同的参数，因此，参数个数为（5×5+1）×6= 156，其中 5×5 为卷积核参数，1 为偏置参数
（3）连接数
卷积后的图像大小为 28×28，因此每个特征图有 28×28 个神经元，每个卷积核参数为（5×5+1）×6，因此，该层的连接数为（5×5+1）×6×28×28=122304
## S2 层（下采样层，也称池化层）：6@14×14
（1）特征图大小
这一层主要是做池化或者特征映射（特征降维），池化单元为 2×2，因此，6 个特征图的大小经池化后即变为 14×14。回顾本文刚开始讲到的池化操作，池化单元之间没有重叠，在池化区域内进行聚合统计后得到新的特征值，因此经 2×2 池化后，每两行两列重新算出一个特征值出来，相当于图像大小减半，因此卷积后的 28×28 图像经 2×2 池化后就变为 14×14。
这一层的计算过程是：2×2 单元里的值相加，然后再乘以训练参数 w，再加上一个偏置参数 b（每一个特征图共享相同的 w 和 b)，然后取 sigmoid 值（S 函数：0-1 区间），作为对应的该单元的值。卷积操作与池化的示意图如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/e618cfd02601f00c561a7565a9c7a32b.png)
（2）参数个数
S2 层由于每个特征图都共享相同的 w 和 b 这两个参数，因此需要 2×6=12 个参数
（3）连接数
下采样之后的图像大小为 14×14，因此 S2 层的每个特征图有 14×14 个神经元，每个池化单元连接数为 2×2+1（1 为偏置量），因此，该层的连接数为（2×2+1）×14×14×6 = 5880
## C3 层（卷积层）：16@10×10
C3 层有 16 个卷积核，卷积模板大小为 5×5。
（1）特征图大小
与 C1 层的分析类似，C3 层的特征图大小为（14-5+1）×（14-5+1）= 10×10
（2）参数个数
需要注意的是，C3 与 S2 并不是全连接而是部分连接，有些是 C3 连接到 S2 三层、有些四层、甚至达到 6 层，通过这种方式提取更多特征，连接的规则如下表所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/eb2c5d0fe70d48328989a921d8c0d354.png)
例如第一列表示 C3 层的第 0 个特征图（feature map）只跟 S2 层的第 0、1 和 2 这三个 feature maps 相连接，计算过程为：用 3 个卷积模板分别与 S2 层的 3 个 feature maps 进行卷积，然后将卷积的结果相加求和，再加上一个偏置，再取 sigmoid 得出卷积后对应的 feature map 了。其它列也是类似（有些是 3 个卷积模板，有些是 4 个，有些是 6 个）。因此，C3 层的参数数目为（5×5×3+1）×6 +（5×5×4+1）×9 +5×5×6+1 = 1516

（3）连接数
卷积后的特征图大小为 10×10，参数数量为 1516，因此连接数为 1516×10×10= 151600
## S4（下采样层，也称池化层）：16@5×5
（1）特征图大小
与 S2 的分析类似，池化单元大小为 2×2，因此，该层与 C3 一样共有 16 个特征图，每个特征图的大小为 5×5。
（2）参数数量
与 S2 的计算类似，所需要参数个数为 16×2 = 32
（3）连接数
连接数为（2×2+1）×5×5×16 = 2000
## C5 层（卷积层）：120
（1）特征图大小
该层有 120 个卷积核，每个卷积核的大小仍为 5×5，因此有 120 个特征图。由于 S4 层的大小为 5×5，而该层的卷积核大小也是 5×5，因此特征图大小为（5-5+1）×（5-5+1）= 1×1。这样该层就刚好变成了全连接，这只是巧合，如果原始输入的图像比较大，则该层就不是全连接了。
（2）参数个数
与前面的分析类似，本层的参数数目为 120×（5×5×16+1） = 48120
（3）连接数
由于该层的特征图大小刚好为 1×1，因此连接数为 48120×1×1=48120
## F6 层（全连接层）：84
1）特征图大小
F6 层有 84 个单元，之所以选这个数字的原因是来自于输出层的设计，对应于一个 7×12 的比特图，如下图所示，-1 表示白色，1 表示黑色，这样每个符号的比特图的黑白色就对应于一个编码。
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/288c57384a469c07dd5ea81d65fa23c1.png)
该层有 84 个特征图，特征图大小与 C5 一样都是 1×1，与 C5 层全连接。
（2）参数个数
由于是全连接，参数数量为（120+1）×84=10164。跟经典神经网络一样，F6 层计算输入向量和权重向量之间的点积，再加上一个偏置，然后将其传递给 sigmoid 函数得出结果。
（3）连接数
由于是全连接，连接数与参数数量一样，也是 10164。
## OUTPUT 层（输出层）：10
Output 层也是全连接层，共有 10 个节点，分别代表数字 0 到 9。如果第 i 个节点的值为 0，则表示网络识别的结果是数字 i。
（1）特征图大小
该层采用径向基函数（RBF）的网络连接方式，假设 x 是上一层的输入，y 是 RBF 的输出，则 RBF 输出的计算方式是：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/7fd3ed11ef60907b1c60af384d8a003a.png)
上式中的 Wij 的值由 i 的比特图编码确定，i 从 0 到 9，j 取值从 0 到 7×12-1。RBF 输出的值越接近于 0，表示当前网络输入的识别结果与字符 i 越接近。

（2）参数个数
由于是全连接，参数个数为 84×10=840
（3）连接数
由于是全连接，连接数与参数个数一样，也是 840

通过以上介绍，已经了解了 LeNet 各层网络的结构、特征图大小、参数数量、连接数量等信息，下图是识别数字 3 的过程，可对照上面介绍各个层的功能进行一一回顾：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/42d35b7f62f644b34707ae01f67d6220.png)
## 编程实现

```
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
#开启tensorflow支持numpy函数，astype是numpy的函数
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
ori_x_test1=x_test

# 将图像从28*28转换成32*32
x_train = tf.pad(x_train, [[0,0], [2,2], [2,2]], mode='constant')
x_test = tf.pad(x_test, [[0,0], [2,2], [2,2]], mode='constant')

# 将像素值缩放到0-1之间
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

# 定义Lenet-5模型
model = models.Sequential([
    # 第一层卷积层，6个卷积核，大小为5*5，使用sigmoid激活函数
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)),
    # 第一层池化层，大小为2*2
    layers.MaxPooling2D((2, 2)),
    # 第二层卷积层，16个卷积核，大小为5*5，使用sigmoid激活函数
    layers.Conv2D(16, (5, 5), activation='relu'),
    # 第二层池化层，大小为2*2
    layers.MaxPooling2D((2, 2)),
    # 第三层卷积层，120个卷积核，大小为5*5，使用sigmoid激活函数
    layers.Conv2D(120, (5, 5), activation='relu'),
    # 将卷积层的输出拉平
    layers.Flatten(),
    # 第一层全连接层，84个节点，使用sigmoid激活函数
    layers.Dense(84, activation='relu'),
    # 输出层，共10个节点，对应0-9十个数字，使用softmax激活函数
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



#取出其中一个测试数据进行测试
testdata = ori_x_test1[100]
testdata = testdata.reshape(-1,28,28)
testdata = tf.pad(testdata, [[0,0], [2,2], [2,2]], mode='constant')
testdata=testdata.reshape(-1, 32, 32, 1)
# 将像素值缩放到0-1之间
testdata = testdata.astype('float32') / 255.0
predictions = model.predict(testdata)
print("预测结果：", np.argmax(predictions))

# 绘制第10个测试数据的图形
plt.imshow(ori_x_test1[100], cmap=plt.cm.binary)
plt.show()
```
输出：
Test loss: 0.03826029598712921
Test accuracy: 0.9879999756813049
预测结果： 6
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/2145254891056f7791343e7516a83cd0.png)

> 参考:https://my.oschina.net/u/876354/blog/1632862

# AlexNet
2012 年，Alex Krizhevsky、Ilya Sutskever 在多伦多大学 Geoff Hinton 的实验室设计出了一个深层的卷积神经网络 AlexNet，夺得了 2012 年 ImageNet LSVRC 的冠军，且准确率远超第二名（top5 错误率为 15.3%，第二名为 26.2%），引起了很大的轰动。AlexNet 可以说是具有历史意义的一个网络结构，在此之前，深度学习已经沉寂了很长时间，自 2012 年 AlexNet 诞生之后，后面的 ImageNet 冠军都是用卷积神经网络（CNN）来做的，并且层次越来越深，使得 CNN 成为在图像识别分类的核心算法模型，带来了深度学习的大爆发。
在本博客之前的文章中已经介绍过了卷积神经网络（CNN）的技术原理（大话卷积神经网络），也回顾过卷积神经网络（CNN）的三个重要特点（大话 CNN 经典模型：LeNet），有兴趣的同学可以打开链接重新回顾一下，在此就不再重复 CNN 基础知识的介绍了。下面将先介绍 AlexNet 的特点，然后再逐层分解解析 AlexNet 网络结构。

## AlexNet 模型的特点
AlexNet 之所以能够成功，跟这个模型设计的特点有关，主要有：

- 使用了非线性激活函数：ReLU
- 防止过拟合的方法：Dropout，数据扩充（Data augmentation）
- 其他：多 GPU 实现，LRN 归一化层的使用

1、使用 ReLU 激活函数
传统的神经网络普遍使用 Sigmoid 或者 tanh 等非线性函数作为激励函数，然而它们容易出现梯度弥散或梯度饱和的情况。以 Sigmoid 函数为例，当输入的值非常大或者非常小的时候，这些神经元的梯度接近于 0（梯度饱和现象），如果输入的初始值很大的话，梯度在反向传播时因为需要乘上一个 Sigmoid 导数，会造成梯度越来越小，导致网络变的很难学习。（详见本公博客的文章：深度学习中常用的激励函数）。
在 AlexNet 中，使用了 ReLU （Rectified Linear Units）激励函数，该函数的公式为：f (x)=max (0,x)，当输入信号 < 0 时，输出都是 0，当输入信号 > 0 时，输出等于输入，如下图所示：

![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/66ab30c567b5435fbae4e38f9da83d74.png)
使用 ReLU 替代 Sigmoid/tanh，由于 ReLU 是线性的，且导数始终为 1，计算量大大减少，收敛速度会比 Sigmoid/tanh 快很多，如下图所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/8884dc923073f3cf4efe6ea662279265.png)
2、数据扩充（Data augmentation）

有一种观点认为神经网络是靠数据喂出来的，如果能够增加训练数据，提供海量数据进行训练，则能够有效提升算法的准确率，因为这样可以避免过拟合，从而可以进一步增大、加深网络结构。而当训练数据有限时，可以通过一些变换从已有的训练数据集中生成一些新的数据，以快速地扩充训练数据。
其中，最简单、通用的图像数据变形的方式：水平翻转图像，从原始图像中随机裁剪、平移变换，颜色、光照变换，如下图所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/d14e7ac31b2f0560b047674d77804395.png)
AlexNet 在训练时，在数据扩充（data augmentation）这样处理：
（1）随机裁剪，对 256×256 的图片进行随机裁剪到 224×224，然后进行水平翻转，相当于将样本数量增加了（（256-224）^2）×2=2048 倍；
（2）测试的时候，对左上、右上、左下、右下、中间分别做了 5 次裁剪，然后翻转，共 10 个裁剪，之后对结果求平均。作者说，如果不做随机裁剪，大网络基本上都过拟合；
（3）对 RGB 空间做 PCA（主成分分析），然后对主成分做一个（0, 0.1）的高斯扰动，也就是对颜色、光照作变换，结果使错误率又下降了 1%。

3、重叠池化 (Overlapping Pooling)
一般的池化（Pooling）是不重叠的，池化区域的窗口大小与步长相同，如下图所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/0c6495716470b88ebf1a33344dd5dac9.png)
在 AlexNet 中使用的池化（Pooling）却是可重叠的，也就是说，在池化的时候，每次移动的步长小于池化的窗口长度。AlexNet 池化的大小为 3×3 的正方形，每次池化移动步长为 2，这样就会出现重叠。重叠池化可以避免过拟合，这个策略贡献了 0.3% 的 Top-5 错误率。
4、局部归一化（Local Response Normalization，简称 LRN）
在神经生物学有一个概念叫做 “侧抑制”（lateral inhibitio），指的是被激活的神经元抑制相邻神经元。归一化（normalization）的目的是 “抑制”，局部归一化就是借鉴了 “侧抑制” 的思想来实现局部抑制，尤其当使用 ReLU 时这种 “侧抑制” 很管用，因为 ReLU 的响应结果是无界的（可以非常大），所以需要归一化。使用局部归一化的方案有助于增加泛化能力。
LRN 的公式如下，核心思想就是利用临近的数据做归一化，这个策略贡献了 1.2% 的 Top-5 错误率。
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/944fcd68193b479c4a5b41aca0029e62.png)
5、Dropout
引入 Dropout 主要是为了防止过拟合。在神经网络中 Dropout 通过修改神经网络本身结构来实现，对于某一层的神经元，通过定义的概率将神经元置为 0，这个神经元就不参与前向和后向传播，就如同在网络中被删除了一样，同时保持输入层与输出层神经元的个数不变，然后按照神经网络的学习方法进行参数更新。在下一次迭代中，又重新随机删除一些神经元（置为 0），直至训练结束。
Dropout 应该算是 AlexNet 中一个很大的创新，以至于 “神经网络之父” Hinton 在后来很长一段时间里的演讲中都拿 Dropout 说事。Dropout 也可以看成是一种模型组合，每次生成的网络结构都不一样，通过组合多个模型的方式能够有效地减少过拟合，Dropout 只需要两倍的训练时间即可实现模型组合（类似取平均）的效果，非常高效。
如下图所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/fd43d2f73e96f4c9e7867d6566241f43.png)
6、多 GPU 训练
AlexNet 当时使用了 GTX580 的 GPU 进行训练，由于单个 GTX 580 GPU 只有 3GB 内存，这限制了在其上训练的网络的最大规模，因此他们在每个 GPU 中放置一半核（或神经元），将网络分布在两个 GPU 上进行并行计算，大大加快了 AlexNet 的训练速度。
## AlexNet 网络结构的逐层解析
下图是 AlexNet 的网络结构图：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/8c544f657baf4fc3e6197af635620d51.png)
AlexNet 网络结构共有 8 层，前面 5 层是卷积层，后面 3 层是全连接层，最后一个全连接层的输出传递给一个 1000 路的 softmax 层，对应 1000 个类标签的分布。
由于 AlexNet 采用了两个 GPU 进行训练，因此，该网络结构图由上下两部分组成，一个 GPU 运行图上方的层，另一个运行图下方的层，两个 GPU 只在特定的层通信。例如第二、四、五层卷积层的核只和同一个 GPU 上的前一层的核特征图相连，第三层卷积层和第二层所有的核特征图相连接，全连接层中的神经元和前一层中的所有神经元相连接。

下面逐层解析 AlexNet 结构：
### 第一层（卷积层）
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/5dac2c7e8560e9bd3b3e1d6181f62627.png)
该层的处理流程为：卷积 -->ReLU--> 池化 --> 归一化，流程图如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/76b05b3179355f8fa5162cd1eae82182.png)
（1）卷积
输入的原始图像大小为 224×224×3（RGB 图像），在训练时会经过预处理变为 227×227×3。在本层使用 96 个 11×11×3 的卷积核进行卷积计算，生成新的像素。由于采用了两个 GPU 并行运算，因此，网络结构图中上下两部分分别承担了 48 个卷积核的运算。
卷积核沿图像按一定的步长往 x 轴方向、y 轴方向移动计算卷积，然后生成新的特征图，其大小为：floor ((img_size - filter_size)/stride) +1 = new_feture_size，其中 floor 表示向下取整，img_size 为图像大小，filter_size 为核大小，stride 为步长，new_feture_size 为卷积后的特征图大小，这个公式表示图像尺寸减去卷积核尺寸除以步长，再加上被减去的核大小像素对应生成的一个像素，结果就是卷积后特征图的大小。
AlexNet 中本层的卷积移动步长是 4 个像素，卷积核经移动计算后生成的特征图大小为 (227-11)/4+1=55，即 55×55。
（2）ReLU
卷积后的 55×55 像素层经过 ReLU 单元的处理，生成激活像素层，尺寸仍为 2 组 55×55×48 的像素层数据。
（3）池化
RuLU 后的像素层再经过池化运算，池化运算的尺寸为 3×3，步长为 2，则池化后图像的尺寸为 (55-3)/2+1=27，即池化后像素的规模为 27×27×96
（4）归一化
池化后的像素层再进行归一化处理，归一化运算的尺寸为 5×5，归一化后的像素规模不变，仍为 27×27×96，这 96 层像素层被分为两组，每组 48 个像素层，分别在一个独立的 GPU 上进行运算。
### 第二层（卷积层）
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/ef1a0047841b5fa5fa66e79d129919c0.png)
该层与第一层类似，处理流程为：卷积 -->ReLU--> 池化 --> 归一化，流程图如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/56bdb3366027bbaf9f856613393c359a.png)
（1）卷积
第二层的输入数据为第一层输出的 27×27×96 的像素层（被分成两组 27×27×48 的像素层放在两个不同 GPU 中进行运算），为方便后续处理，在这里每幅像素层的上下左右边缘都被填充了 2 个像素（填充 0），即图像的大小变为 (27+2+2) ×(27+2+2)。第二层的卷积核大小为 5×5，移动步长为 1 个像素，跟第一层第（1）点的计算公式一样，经卷积核计算后的像素层大小变为 (27+2+2-5)/1+1=27，即卷积后大小为 27×27。
本层使用了 256 个 5×5×48 的卷积核，同样也是被分成两组，每组为 128 个，分给两个 GPU 进行卷积运算，结果生成两组 27×27×128 个卷积后的像素层。
（2）ReLU
这些像素层经过 ReLU 单元的处理，生成激活像素层，尺寸仍为两组 27×27×128 的像素层。
（3）池化
再经过池化运算的处理，池化运算的尺寸为 3×3，步长为 2，池化后图像的尺寸为 (57-3)/2+1=13，即池化后像素的规模为 2 组 13×13×128 的像素层
（4）归一化
然后再经归一化处理，归一化运算的尺度为 5×5，归一化后的像素层的规模为 2 组 13×13×128 的像素层，分别由 2 个 GPU 进行运算。
### 第三层（卷积层）
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/80748fc3cc58e6f278bdbd4444d62a25.png)
第三层的处理流程为：卷积 -->ReLU
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/d5a08fb7ed3281b15b020e20c71e0088.png)
（1）卷积
第三层输入数据为第二层输出的 2 组 13×13×128 的像素层，为便于后续处理，每幅像素层的上下左右边缘都填充 1 个像素，填充后变为 (13+1+1)×(13+1+1)×128，分布在两个 GPU 中进行运算。
这一层中每个 GPU 都有 192 个卷积核，每个卷积核的尺寸是 3×3×256。因此，每个 GPU 中的卷积核都能对 2 组 13×13×128 的像素层的所有数据进行卷积运算。如该层的结构图所示，两个 GPU 有通过交叉的虚线连接，也就是说每个 GPU 要处理来自前一层的所有 GPU 的输入。
本层卷积的步长是 1 个像素，经过卷积运算后的尺寸为 (13+1+1-3)/1+1=13，即每个 GPU 中共 13×13×192 个卷积核，2 个 GPU 中共有 13×13×384 个卷积后的像素层。
（2）ReLU
卷积后的像素层经过 ReLU 单元的处理，生成激活像素层，尺寸仍为 2 组 13×13×192 的像素层，分配给两组 GPU 处理。
### 第四层（卷积层）
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/0866c80c2c9216a1023a0aa172e667dd.png)
与第三层类似，第四层的处理流程为：卷积 -->ReLU
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/ab9675459db46906f6aa3ad25bd1f95e.png)
1）卷积
第四层输入数据为第三层输出的 2 组 13×13×192 的像素层，类似于第三层，为便于后续处理，每幅像素层的上下左右边缘都填充 1 个像素，填充后的尺寸变为 (13+1+1)×(13+1+1)×192，分布在两个 GPU 中进行运算。
这一层中每个 GPU 都有 192 个卷积核，每个卷积核的尺寸是 3×3×192（与第三层不同，第四层的 GPU 之间没有虚线连接，也即 GPU 之间没有通信）。卷积的移动步长是 1 个像素，经卷积运算后的尺寸为 (13+1+1-3)/1+1=13，每个 GPU 中有 13×13×192 个卷积核，2 个 GPU 卷积后生成 13×13×384 的像素层。
（2）ReLU
卷积后的像素层经过 ReLU 单元处理，生成激活像素层，尺寸仍为 2 组 13×13×192 像素层，分配给两个 GPU 处理。
### 第五层（卷积层）
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/f776bccde1428f70d4a71fe79685c929.png)
第五层的处理流程为：卷积 -->ReLU--> 池化
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/c833ddb29f3b044c83299a4d2c691bab.png)
（1）卷积
第五层输入数据为第四层输出的 2 组 13×13×192 的像素层，为便于后续处理，每幅像素层的上下左右边缘都填充 1 个像素，填充后的尺寸变为 (13+1+1)×(13+1+1) ，2 组像素层数据被送至 2 个不同的 GPU 中进行运算。
这一层中每个 GPU 都有 128 个卷积核，每个卷积核的尺寸是 3×3×192，卷积的步长是 1 个像素，经卷积后的尺寸为 (13+1+1-3)/1+1=13，每个 GPU 中有 13×13×128 个卷积核，2 个 GPU 卷积后生成 13×13×256 的像素层。
（2）ReLU
卷积后的像素层经过 ReLU 单元处理，生成激活像素层，尺寸仍为 2 组 13×13×128 像素层，由两个 GPU 分别处理。
（3）池化
2 组 13×13×128 像素层分别在 2 个不同 GPU 中进行池化运算处理，池化运算的尺寸为 3×3，步长为 2，池化后图像的尺寸为 (13-3)/2+1=6，即池化后像素的规模为两组 6×6×128 的像素层数据，共有 6×6×256 的像素层数据。
### 第六层（全连接层）
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/fbe254ee242a6cad7411b2155410088e.png)
第六层的处理流程为：卷积（全连接）-->ReLU-->Dropout
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/74cc588db43c2fd708f55fe19d147399.png)
（1）卷积（全连接）
第六层输入数据是第五层的输出，尺寸为 6×6×256。本层共有 4096 个卷积核，每个卷积核的尺寸为 6×6×256，由于卷积核的尺寸刚好与待处理特征图（输入）的尺寸相同，即卷积核中的每个系数只与特征图（输入）尺寸的一个像素值相乘，一一对应，因此，该层被称为全连接层。由于卷积核与特征图的尺寸相同，卷积运算后只有一个值，因此，卷积后的像素层尺寸为 4096×1×1，即有 4096 个神经元。
（2）ReLU
这 4096 个运算结果通过 ReLU 激活函数生成 4096 个值。
（3）Dropout
然后再通过 Dropout 运算，输出 4096 个结果值。
### 第七层（全连接层）
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/63ebc77a6b744ff514a4c607d71757af.png)
第七层的处理流程为：全连接 -->ReLU-->Dropout
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/1f7442e6ead5d4b4c7d4a52248b232be.png)
第六层输出的 4096 个数据与第七层的 4096 个神经元进行全连接，然后经 ReLU 进行处理后生成 4096 个数据，再经过 Dropout 处理后输出 4096 个数据。
### 第八层（全连接层）
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/3e20f0e87c49db6502f427ecbc75431e.png)
第八层的处理流程为：全连接
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/3c0eacd7aac19ae5489170084dce1362.png)
第七层输出的 4096 个数据与第八层的 1000 个神经元进行全连接，经过训练后输出 1000 个 float 型的值，这就是预测结果。

以上就是关于 AlexNet 网络结构图的逐层解析了，看起来挺复杂的，下面是一个简图，看起来就清爽很多啊
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/f4f1a34afd3fc94f4414af11d4e64222.png)
通过前面的介绍，可以看出 AlexNet 的特点和创新之处，主要如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/5483e017be2787ac351fe2a50b4bfc1d.png)
##  编程实现
下载imagenet数据集，
Keras提供的keras.datasets模块可以用来直接加载ImageNet数据集。不过需要注意的是，ImageNet数据集非常大，包含数百万张高分辨率图像，因此通常需要使用分布式计算或者在GPU上进行训练。

CIFAR-10数据集是一个常用的图像分类数据集，包含10个类别的图像，每个类别包含6000张32x32像素的彩色图像，总共60000张，其中50000张是用于训练，10000张是用于测试。这10个类别分别是：

0. 飞机（airplane）
1. 汽车（automobile）
2. 鸟类（bird）
3. 猫（cat）
4. 鹿（deer）
5. 狗（dog）
6. 青蛙（frog）
7. 马（horse）
8. 船（ship）
9. 卡车（truck）

每个图像的标签是一个0到9之间的整数，对应上述10个类别中的一个。因此，我们可以使用这些标签来训练和测试图像分类模型。
你可以使用以下代码来加载这个小样本数据集：

```
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
```
执行后，日志里有一直在下载的过程，下载很慢，路径
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
我们可以手动下载下来，重命名为：cifar-10-batches-py.tar.gz，然后上传到 ~/.keras/datasets目录即可（不用解压），程序会离线解压该文件，window下是：C:\Users\你的用户\.keras\datasets
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/36b1c681ccca13d6e68cbe7365292621.png)
再次运行输出
(50000, 32, 32, 3)
随机加载100张，看看效果

```
# 随机选择100张图片进行显示
indices = np.random.choice(len(x_train), size=100, replace=False)
images = x_train[indices]
labels = y_train[indices]

# 绘制图片
fig = plt.figure(figsize=(10, 10))
for i in range(10):
    for j in range(10):
        index = i * 10 + j
        ax = fig.add_subplot(10, 10, index + 1)
        ax.imshow(images[index])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(labels[index][0])
plt.show()
```
显示
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/ec057edcc7abae90de58a5efbc7fe970.png)
因为数据集总共有6万张，格式32*32，使用alexnet模型进行计算，图像需要转换224*224，rgb通道数3，每个像素都需要转换成float32，这样导致数gpu显存占用过大导致内存溢出，
需要占用显存=60000*224*224*3*4＞＝３０ＧＢ，
所以需增量式进行训练

```
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
"""
在Python中，我们可以使用TensorFlow或Keras等深度学习框架来加载CIFAR-10数据集。为了有效地处理大量图像数据，我们可以使用生成器函数和yield语句来逐批加载数据。
生成器函数是一个Python函数，它使用yield语句来产生一个序列的值。当函数执行到yield语句时，它会将当前的值返回给调用者，并暂停函数的执行。当函数再次被调用时，它会从上一次暂停的位置继续执行，并返回下一个值。
"""
def cifar10_generator(x, y, batch_size):
    """
    CIFAR-10 data generator.
    """
    while True:
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            x_batch = tf.image.resize_with_pad(x_batch, target_height=224, target_width=224)
            x_batch = x_batch.astype('float32') / 255.0
            yield x_batch, y_batch

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def alexnet(input_shape, num_classes):
    model = tf.keras.Sequential([
        Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(256, (5,5), strides=(1,1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu'),
        Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu'),
        Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# 定义一些超参数
batch_size = 256
epochs = 5
learning_rate = 0.001

# 定义生成器
train_generator = cifar10_generator(x_train, y_train, batch_size)
test_generator = cifar10_generator(x_test, y_test, batch_size)

# 定义模型
input_shape = (224,224,3)
num_classes = 10
model = alexnet(input_shape, num_classes)

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 定义 ModelCheckpoint 回调函数
checkpoint = tf.keras.callbacks.ModelCheckpoint('./AlexNet.h5', save_best_only=True, save_weights_only=False, monitor='val_loss')

# 训练模型
model.fit(train_generator,
          epochs=epochs,
          steps_per_epoch=len(x_train)//batch_size,
          validation_data=test_generator,
          validation_steps=len(x_test)//batch_size,
          callbacks=[checkpoint]
          )
test_loss, test_acc = model.evaluate(test_generator, y_test)
print('Test accuracy:', test_acc)
```
预测结果和显示图像

```
# 在这里添加您的识别代码
model = tf.keras.models.load_model('./AlexNet.h5')
srcImage=x_test[105]
p_test=np.array([srcImage])
p_test = tf.image.resize_with_pad(p_test, target_height=224, target_width=224)
p_test = p_test.astype('float32') / 255.0
predictions = model.predict(p_test)
print("识别结果为：" + str(np.argmax(predictions)))
# 绘制第10个测试数据的图形
plt.imshow(srcImage, cmap=plt.cm.binary)
plt.show()
```
输出：1
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/332050ef4034e899a28f34faf4a825e8.png)


> 参考:https://my.oschina.net/u/876354/blog/1633143
# VGGNet
2014 年，牛津大学计算机视觉组（Visual Geometry Group）和 Google DeepMind 公司的研究员一起研发出了新的深度卷积神经网络：VGGNet，并取得了 ILSVRC2014 比赛分类项目的第二名（第一名是 GoogLeNet，也是同年提出的）和定位项目的第一名。
VGGNet 探索了卷积神经网络的深度与其性能之间的关系，成功地构筑了 16~19 层深的卷积神经网络，证明了增加网络的深度能够在一定程度上影响网络最终的性能，使错误率大幅下降，同时拓展性又很强，迁移到其它图片数据上的泛化性也非常好。到目前为止，VGG 仍然被用来提取图像特征。
VGGNet 可以看成是加深版本的 AlexNet，都是由卷积层、全连接层两大部分构成。
## VGG 的特点
先看一下 VGG 的结构图
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/d252a3f3ac84945b426d262138b364fb.png)
1、结构简洁
VGG 由 5 层卷积层、3 层全连接层、softmax 输出层构成，层与层之间使用 max-pooling（最大化池）分开，所有隐层的激活单元都采用 ReLU 函数。
2、小卷积核和多卷积子层
VGG 使用多个较小卷积核（3x3）的卷积层代替一个卷积核较大的卷积层，一方面可以减少参数，另一方面相当于进行了更多的非线性映射，可以增加网络的拟合 / 表达能力。
小卷积核是 VGG 的一个重要特点，虽然 VGG 是在模仿 AlexNet 的网络结构，但没有采用 AlexNet 中比较大的卷积核尺寸（如 7x7），而是通过降低卷积核的大小（3x3），增加卷积子层数来达到同样的性能（VGG：从 1 到 4 卷积子层，AlexNet：1 子层）。
VGG 的作者认为两个 3x3 的卷积堆叠获得的感受野大小，相当一个 5x5 的卷积；而 3 个 3x3 卷积的堆叠获取到的感受野相当于一个 7x7 的卷积。这样可以增加非线性映射，也能很好地减少参数（例如 7x7 的参数为 49 个，而 3 个 3x3 的参数为 27），如下图所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/b3dc987629ffb5a11d5b73961a91aa9f.png)
3、小池化核
相比 AlexNet 的 3x3 的池化核，VGG 全部采用 2x2 的池化核。
4、通道数多
VGG 网络第一层的通道数为 64，后面每层都进行了翻倍，最多到 512 个通道，通道数的增加，使得更多的信息可以被提取出来。
5、层数更深、特征图更宽
由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，控制了计算量的增加规模。
6、全连接转卷积（测试阶段）
这也是 VGG 的一个特点，在网络测试阶段将训练阶段的三个全连接替换为三个卷积，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意宽或高为的输入，这在测试阶段很重要。
如本节第一个图所示，输入图像是 224x224x3，如果后面三个层都是全连接，那么在测试阶段就只能将测试的图像全部都要缩放大小到 224x224x3，才能符合后面全连接层的输入数量要求，这样就不便于测试工作的开展。
而 “全连接转卷积”，替换过程如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/9dd75009d679e7ec93a92d1e3f1575c9.png)
例如 7x7x512 的层要跟 4096 个神经元的层做全连接，则替换为对 7x7x512 的层作通道数为 4096、卷积核为 1x1 的卷积。
这个 “全连接转卷积” 的思路是 VGG 作者参考了 OverFeat 的工作思路，例如下图是 OverFeat 将全连接换成卷积后，则可以来处理任意分辨率（在整张图）上计算卷积，这就是无需对原图做重新缩放处理的优势。
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/9be9abe7d1198f7809c722b748cedf92.png)
## VGG 的网络结构
下图是来自论文《Very Deep Convolutional Networks for Large-Scale Image Recognition》（基于甚深层卷积网络的大规模图像识别）的 VGG 网络结构，正是在这篇论文中提出了 VGG，如下图：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/dfb97c18fac5d82766055d9bd70a79fa.png)
在这篇论文中分别使用了 A、A-LRN、B、C、D、E 这 6 种网络结构进行测试，这 6 种网络结构相似，都是由 5 层卷积层、3 层全连接层组成，其中区别在于每个卷积层的子层数量不同，从 A 至 E 依次增加（子层数量从 1 到 4），总的网络深度从 11 层到 19 层（添加的层以粗体显示），表格中的卷积层参数表示为 “conv⟨感受野大小⟩- 通道数⟩”，例如 con3-128，表示使用 3x3 的卷积核，通道数为 128。为了简洁起见，在表格中不显示 ReLU 激活功能。
其中，网络结构 D 就是著名的 VGG16，网络结构 E 就是著名的 VGG19。

以网络结构 D（VGG16）为例，介绍其处理过程如下，请对比上面的表格和下方这张图，留意图中的数字变化，有助于理解 VGG16 的处理过程：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/df2c7cc6443b6b99bdb1d1f7a4bc1886.png)
1、输入 224x224x3 的图片，经 64 个 3x3 的卷积核作两次卷积 + ReLU，卷积后的尺寸变为 224x224x64
2、作 max pooling（最大化池化），池化单元尺寸为 2x2（效果为图像尺寸减半），池化后的尺寸变为 112x112x64
3、经 128 个 3x3 的卷积核作两次卷积 + ReLU，尺寸变为 112x112x128
4、作 2x2 的 max pooling 池化，尺寸变为 56x56x128
5、经 256 个 3x3 的卷积核作三次卷积 + ReLU，尺寸变为 56x56x256
6、作 2x2 的 max pooling 池化，尺寸变为 28x28x256
7、经 512 个 3x3 的卷积核作三次卷积 + ReLU，尺寸变为 28x28x512
8、作 2x2 的 max pooling 池化，尺寸变为 14x14x512
9、经 512 个 3x3 的卷积核作三次卷积 + ReLU，尺寸变为 14x14x512
10、作 2x2 的 max pooling 池化，尺寸变为 7x7x512
11、与两层 1x1x4096，一层 1x1x1000 进行全连接 + ReLU（共三层）
12、通过 softmax 输出 1000 个预测结果

以上就是 VGG16（网络结构 D）各层的处理过程，A、A-LRN、B、C、E 其它网络结构的处理过程也是类似，执行过程如下（以 VGG16 为例）：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/baafe0ddad3c6d64add31454281c8a9e.png)
从上面的过程可以看出 VGG 网络结构还是挺简洁的，都是由小卷积核、小池化核、ReLU 组合而成。其简化图如下（以 VGG16 为例）：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/32713e886187406e7c8a7e3f555ee552.png)
A、A-LRN、B、C、D、E 这 6 种网络结构的深度虽然从 11 层增加至 19 层，但参数量变化不大，这是由于基本上都是采用了小卷积核（3x3，只有 9 个参数），这 6 种结构的参数数量（百万级）并未发生太大变化，这是因为在网络中，参数主要集中在全连接层。
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/35a5e47c69cd51e37d00abcced080c9d.png)
经作者对 A、A-LRN、B、C、D、E 这 6 种网络结构进行单尺度的评估，错误率结果如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/74f7b0832f63b852bdd335662359a8fd.png)
从上表可以看出：  
**1、LRN 层无性能增益（A-LRN）**  
VGG 作者通过网络 A-LRN 发现，AlexNet 曾经用到的 LRN 层（local response normalization，局部响应归一化）并没有带来性能的提升，因此在其它组的网络中均没再出现 LRN 层。  
**2、随着深度增加，分类性能逐渐提高（A、B、C、D、E）**  
从 11 层的 A 到 19 层的 E，网络深度增加对 top1 和 top5 的错误率下降很明显。  
**3、多个小卷积核比单个大卷积核性能好（B）**  
VGG 作者做了实验用 B 和自己一个不在实验组里的较浅网络比较，较浅网络用 conv5x5 来代替 B 的两个 conv3x3，结果显示多个小卷积核比单个大卷积核效果要好。

最后进行个小结：  
**1、通过增加深度能有效地提升性能；  
2、最佳模型：VGG16，从头到尾只有 3x3 卷积与 2x2 池化，简洁优美；  
3、卷积可代替全连接，可适应各种尺寸的图片**

## 编程实现
ILSVRC2014 数据集在image-net下载目前需要注册，并且需要审批比较麻烦，可以在阿里云天池数据集上下载ILSVRC2017版本（可以使用钉钉或者支付宝实名认证登录下，很多大型数据集都可以登录后直接下载），地址：https://tianchi.aliyun.com/dataset/92252，下载imagenet_object_localization_patched2019 (1).tar.gz，数据集大小155GB
由于数据集过大，我这里依然使用cifar10

>VGGNet和AlexNet都是深度神经网络模型，VGGNet比AlexNet更深，因此它需要更多的计算资源和时间来训练。具体来说，VGGNet有16层或19层，而AlexNet只有8层。这意味着VGGNet需要处理更多的参数和数据，需要更长的训练时间。此外，VGGNet使用了更小的卷积核，这也导致了更多的计算量。所以，VGGNet训练比AlexNet慢很多是很正常的。
>
```
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

np_config.enable_numpy_behavior()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def cifar10_generator(x, y, batch_size):
    while True:
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            x_batch = tf.image.resize_with_pad(x_batch, target_height=224, target_width=224)
            x_batch = x_batch.astype('float32') / 255.0
            yield x_batch, y_batch


def vggnet(input_shape, num_classes):
    # 定义VGGNet
    model = Sequential([
        # 第一层卷积和池化
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 第二层卷积和池化
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 第三层卷积和池化
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 第四层卷积和池化
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 第五层卷积和池化
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 将输出的特征图展平，并连接全连接层
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(10, activation='softmax')
    ])

    return model

# 定义一些超参数
batch_size = 128
epochs = 5
learning_rate = 0.001

# 定义生成器
train_generator = cifar10_generator(x_train, y_train, batch_size)
test_generator = cifar10_generator(x_test, y_test, batch_size)

# 定义模型
input_shape = (224,224,3)
num_classes = 10
model = vggnet(input_shape, num_classes)
model.summary()
# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 定义 ModelCheckpoint 回调函数
checkpoint = tf.keras.callbacks.ModelCheckpoint('./VGGNet.h5', save_best_only=True, save_weights_only=False, monitor='val_loss')

# 训练模型
model.fit(train_generator,
          epochs=epochs,
          steps_per_epoch=len(x_train)//batch_size,
          validation_data=test_generator,
          validation_steps=len(x_test)//batch_size,
          callbacks=[checkpoint]
          )
test_loss, test_acc = model.evaluate(test_generator, y_test)
print('Test accuracy:', test_acc)



```

> 参考:https://my.oschina.net/u/876354/blog/1634322
# GoogLeNet
2014 年，GoogLeNet 和 VGG 是当年 ImageNet 挑战赛 (ILSVRC14) 的双雄，GoogLeNet 获得了第一名、VGG 获得了第二名，这两类模型结构的共同特点是层次更深了。VGG 继承了 LeNet 以及 AlexNet 的一些框架结构，而 GoogLeNet 则做了更加大胆的网络结构尝试，虽然深度只有 22 层，但大小却比 AlexNet 和 VGG 小很多，GoogleNet 参数为 500 万个，AlexNet 参数个数是 GoogleNet 的 12 倍，VGGNet 参数又是 AlexNet 的 3 倍，因此在内存或计算资源有限时，GoogleNet 是比较好的选择；从模型结果来看，GoogLeNet 的性能却更加优越。

小知识：GoogLeNet 是谷歌（Google）研究出来的深度网络结构，为什么不叫 “GoogleNet”，而叫 “GoogLeNet”，据说是为了向 “LeNet” 致敬，因此取名为 “GoogLeNet”

那么，GoogLeNet 是如何进一步提升性能的呢？  
一般来说，提升网络性能最直接的办法就是增加网络深度和宽度，深度指网络层次数量、宽度指神经元数量。但这种方式存在以下问题：  
（1）参数太多，如果训练数据集有限，很容易产生过拟合；  
（2）网络越大、参数越多，计算复杂度越大，难以应用；  
（3）网络越深，容易出现梯度弥散问题（梯度越往后穿越容易消失），难以优化模型。  
所以，有人调侃 “深度学习” 其实是 “深度调参”。  
解决这些问题的方法当然就是在增加网络深度和宽度的同时减少参数，为了减少参数，自然就想到将全连接变成稀疏连接。但是在实现上，全连接变成稀疏连接后实际计算量并不会有质的提升，因为大部分硬件是针对密集矩阵计算优化的，稀疏矩阵虽然数据量少，但是计算所消耗的时间却很难减少。

那么，有没有一种方法既能保持网络结构的稀疏性，又能利用密集矩阵的高计算性能。大量的文献表明可以将稀疏矩阵聚类为较为密集的子矩阵来提高计算性能，就如人类的大脑是可以看做是神经元的重复堆积，因此，GoogLeNet 团队提出了 Inception 网络结构，就是构造一种 “基础神经元” 结构，来搭建一个稀疏性、高计算性能的网络结构。
【问题来了】什么是 Inception 呢？
Inception 历经了 V1、V2、V3、V4 等多个版本的发展，不断趋于完善，下面一一进行介绍
## Inception V1
通过设计一个稀疏网络结构，但是能够产生稠密的数据，既能增加神经网络表现，又能保证计算资源的使用效率。谷歌提出了最原始 Inception 的基本结构：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/1a4b5804c3a969cfab032e43f029607c.png)
该结构将 CNN 中常用的卷积（1x1，3x3，5x5）、池化操作（3x3）堆叠在一起（卷积、池化后的尺寸相同，将通道相加），一方面增加了网络的宽度，另一方面也增加了网络对尺度的适应性。
网络卷积层中的网络能够提取输入的每一个细节信息，同时 5x5 的滤波器也能够覆盖大部分接受层的的输入。还可以进行一个池化操作，以减少空间大小，降低过度拟合。在这些层之上，在每一个卷积层后都要做一个 ReLU 操作，以增加网络的非线性特征。
然而这个 Inception 原始版本，所有的卷积核都在上一层的所有输出上来做，而那个 5x5 的卷积核所需的计算量就太大了，造成了特征图的厚度很大，为了避免这种情况，在 3x3 前、5x5 前、max pooling 后分别加上了 1x1 的卷积核，以起到了降低特征图厚度的作用，这也就形成了 Inception v1 的网络结构，如下图所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/56c17d869d516346212da2dfe7dcdfba.png)
1x1 的卷积核有什么用呢？
1x1 卷积的主要目的是为了减少维度，还用于修正线性激活（ReLU）。比如，上一层的输出为 100x100x128，经过具有 256 个通道的 5x5 卷积层之后 (stride=1，pad=2)，输出数据为 100x100x256，其中，卷积层的参数为 128x5x5x256= 819200。而假如上一层输出先经过具有 32 个通道的 1x1 卷积层，再经过具有 256 个输出的 5x5 卷积层，那么输出数据仍为为 100x100x256，但卷积参数量已经减少为 128x1x1x32 + 32x5x5x256= 204800，大约减少了 4 倍。

基于 Inception 构建了 GoogLeNet 的网络结构如下（共 22 层）：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/57e2088c8a62e87160fba0adca6bf079.png)
对上图说明如下：  
（1）GoogLeNet 采用了模块化的结构（Inception 结构），方便增添和修改；  
（2）网络最后采用了 average pooling（平均池化）来代替全连接层，该想法来自 NIN（Network in Network），事实证明这样可以将准确率提高 0.6%。但是，实际在最后还是加了一个全连接层，主要是为了方便对输出进行灵活调整；  
（3）虽然移除了全连接，但是网络中依然使用了 Dropout ;   
（4）为了避免梯度消失，网络额外增加了 2 个辅助的 softmax 用于向前传导梯度（辅助分类器）。辅助分类器是将中间某一层的输出用作分类，并按一个较小的权重（0.3）加到最终分类结果中，这样相当于做了模型融合，同时给网络增加了反向传播的梯度信号，也提供了额外的正则化，对于整个网络的训练很有裨益。而在实际测试的时候，这两个额外的 softmax 会被去掉。

GoogLeNet 的网络结构图细节如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/cf571825df826fe319bd8811fdbd661c.png)
注：上表中的 “#3x3 reduce”，“#5x5 reduce” 表示在 3x3，5x5 卷积操作之前使用了 1x1 卷积的数量。

GoogLeNet 网络结构明细表解析如下：  
**0、输入**  
原始输入图像为 224x224x3，且都进行了零均值化的预处理操作（图像每个像素减去均值）。  
**1、第一层（卷积层）**  
使用 7x7 的卷积核（滑动步长 2，padding 为 3），64 通道，输出为 112x112x64，卷积后进行 ReLU 操作  
经过 3x3 的 max pooling（步长为 2），输出为 ((112 - 3+1)/2)+1=56，即 56x56x64，再进行 ReLU 操作  
**2、第二层（卷积层）**  
使用 3x3 的卷积核（滑动步长为 1，padding 为 1），192 通道，输出为 56x56x192，卷积后进行 ReLU 操作  
经过 3x3 的 max pooling（步长为 2），输出为 ((56 - 3+1)/2)+1=28，即 28x28x192，再进行 ReLU 操作  
**3a、第三层（Inception 3a 层）**  
分为四个分支，采用不同尺度的卷积核来进行处理  
（1）64 个 1x1 的卷积核，然后 RuLU，输出 28x28x64  
（2）96 个 1x1 的卷积核，作为 3x3 卷积核之前的降维，变成 28x28x96，然后进行 ReLU 计算，再进行 128 个 3x3 的卷积（padding 为 1），输出 28x28x128  
（3）16 个 1x1 的卷积核，作为 5x5 卷积核之前的降维，变成 28x28x16，进行 ReLU 计算后，再进行 32 个 5x5 的卷积（padding 为 2），输出 28x28x32  
（4）pool 层，使用 3x3 的核（padding 为 1），输出 28x28x192，然后进行 32 个 1x1 的卷积，输出 28x28x32。  
将四个结果进行连接，对这四部分输出结果的第三维并联，即 64+128+32+32=256，最终输出 28x28x256  
**3b、第三层（Inception 3b 层）**  
（1）128 个 1x1 的卷积核，然后 RuLU，输出 28x28x128  
（2）128 个 1x1 的卷积核，作为 3x3 卷积核之前的降维，变成 28x28x128，进行 ReLU，再进行 192 个 3x3 的卷积（padding 为 1），输出 28x28x192  
（3）32 个 1x1 的卷积核，作为 5x5 卷积核之前的降维，变成 28x28x32，进行 ReLU 计算后，再进行 96 个 5x5 的卷积（padding 为 2），输出 28x28x96  
（4）pool 层，使用 3x3 的核（padding 为 1），输出 28x28x256，然后进行 64 个 1x1 的卷积，输出 28x28x64。  
将四个结果进行连接，对这四部分输出结果的第三维并联，即 128+192+96+64=480，最终输出输出为 28x28x480

第四层（4a,4b,4c,4d,4e）、第五层（5a,5b）……，与 3a、3b 类似，在此就不再重复。

从 GoogLeNet 的实验结果来看，效果很明显，差错率比 MSRA、VGG 等模型都要低，对比结果如下表所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/ee261d58d84ee801f2877b6454887bb5.png)
## Inception V2
GoogLeNet 凭借其优秀的表现，得到了很多研究人员的学习和使用，因此 GoogLeNet 团队又对其进行了进一步地发掘改进，产生了升级版本的 GoogLeNet。  
GoogLeNet 设计的初衷就是要又准又快，而如果只是单纯的堆叠网络虽然可以提高准确率，但是会导致计算效率有明显的下降，所以如何在不增加过多计算量的同时提高网络的表达能力就成为了一个问题。  
Inception V2 版本的解决方案就是修改 Inception 的内部计算逻辑，提出了比较特殊的 “卷积” 计算结构。

**1、卷积分解（Factorizing Convolutions）**  
大尺寸的卷积核可以带来更大的感受野，但也意味着会产生更多的参数，比如 5x5 卷积核的参数有 25 个，3x3 卷积核的参数有 9 个，前者是后者的 25/9=2.78 倍。因此，GoogLeNet 团队提出可以用 2 个连续的 3x3 卷积层组成的小网络来代替单个的 5x5 卷积层，即在保持感受野范围的同时又减少了参数量，如下图：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/e0870940855835a40674f1939da89acd.png)
那么这种替代方案会造成表达能力的下降吗？通过大量实验表明，并不会造成表达缺失。
可以看出，大卷积核完全可以由一系列的 3x3 卷积核来替代，那能不能再分解得更小一点呢？GoogLeNet 团队考虑了 nx1 的卷积核，如下图所示，用 3 个 3x1 取代 3x3 卷积：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/429a3c916d77949c256f2476d1407117.png)
因此，任意 nxn 的卷积都可以通过 1xn 卷积后接 nx1 卷积来替代。GoogLeNet 团队发现在网络的前期使用这种分解效果并不好，在中度大小的特征图（feature map）上使用效果才会更好（特征图大小建议在 12 到 20 之间）。
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/fa38249c67bc296fde3d8e0085cdf8bd.png)
**2、降低特征图大小**  
一般情况下，如果想让图像缩小，可以有如下两种方式：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/45957aa32edd7a5a63b5014fa28265d1.png)
先池化再作 Inception 卷积，或者先作 Inception 卷积再作池化。但是方法一（左图）先作 pooling（池化）会导致特征表示遇到瓶颈（特征缺失），方法二（右图）是正常的缩小，但计算量很大。为了同时保持特征表示且降低计算量，将网络结构改为下图，使用两个并行化的模块来降低计算量（卷积、池化并行执行，再进行合并）
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/1652ae3a70555a299b7a59cdd443e221.png)
使用 Inception V2 作改进版的 GoogLeNet，网络结构图如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/3c60015e23aedf823080bcd95fdfa9d2.png)
注：上表中的 Figure 5 指没有进化的 Inception，Figure 6 是指小卷积版的 Inception（用 3x3 卷积核代替 5x5 卷积核），Figure 7 是指不对称版的 Inception（用 1xn、nx1 卷积核代替 nxn 卷积核）。

经实验，模型结果与旧的 GoogleNet 相比有较大提升，如下表所示：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/fd88fa8c32b3198bfb09a3eb14764b7d.png)
## Inception V3
Inception V3 一个最重要的改进是分解（Factorization），将 7x7 分解成两个一维的卷积（1x7,7x1），3x3 也是一样（1x3,3x1），这样的好处，既可以加速计算，又可以将 1 个卷积拆成 2 个卷积，使得网络深度进一步增加，增加了网络的非线性（每增加一层都要进行 ReLU）。
另外，网络输入从 224x224 变为了 299x299。
## Inception V4
Inception V4 研究了 Inception 模块与残差连接的结合。ResNet 结构大大地加深了网络深度，还极大地提升了训练速度，同时性能也有提升（ResNet 的技术原理介绍见本博客之前的文章：大话深度残差网络 ResNet）。
Inception V4 主要利用残差连接（Residual Connection）来改进 V3 结构，得到 Inception-ResNet-v1，Inception-ResNet-v2，Inception-v4 网络。
ResNet 的残差结构如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/a6822bd8e2fdf590d972a211618706e6.png)
将该结构与 Inception 相结合，变成下图：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/b876dceb5703dcc997bba914e4935a76.png)
通过 20 个类似的模块组合，Inception-ResNet 构建如下：
![在这里插入图片描述](/docs/images/content/programming/ai/deep_learning/cnn/dl_04_cnn_models.md.images/a0f335f834199e207f579718c3c31f22.png)
## 编程实现
后续补充

> 参考:https://my.oschina.net/u/876354/blog/1637819