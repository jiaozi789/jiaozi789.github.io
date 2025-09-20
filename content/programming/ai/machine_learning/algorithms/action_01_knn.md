---
title: "机器学习实战教程（一）：K-近邻（KNN）算法"
date: 2025-09-18T16:55:17+08:00
weight: 1
# bookComments: false
# bookSearchExclude: false
---

### 一、简单k-近邻算法

本文将从k-近邻算法的思想开始讲起，使用python3一步一步编写代码进行实战训练。并且，我也提供了相应的数据集，对代码进行了详细的注释。除此之外，本文也对sklearn实现k-近邻算法的方法进行了讲解。实战实例：电影类别分类、约会网站配对效果判定、手写数字识别。

  文章中大部分文字和例题参考自[https://cuijiahua.com/blog/2017/11/ml\_1\_knn.html](https://cuijiahua.com/blog/2017/11/ml_1_knn.html)  对原文很多代码进行了简化  
  感谢这篇文章加速本人入门速度

#### 1、k-近邻法简介

k近邻法(k-nearest neighbor, k-NN)是1967年由Cover T和Hart P提出的一种基本分类与回归方法。它的工作原理是：存在一个样本数据集合，也称作为训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一个数据与所属分类的对应关系。输入没有标签的新数据后，将新的数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本最相似数据(最近邻)的分类标签。一般来说，我们只选择样本数据集中前k个最相似的数据，这就是k-近邻算法中k的出处，通常k是不大于20的整数。最后，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。

举个简单的例子，我们可以使用k-近邻算法分类一个电影是爱情片还是动作片。

![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/75feb6140ac91f92c94ee92c5d51ded2.png)

表1.1 每部电影的打斗镜头数、接吻镜头数以及电影类型

表1.1 就是我们已有的数据集合，也就是训练样本集。这个数据集有两个特征，即打斗镜头数和接吻镜头数。除此之外，我们也知道每个电影的所属类型，即分类标签。用肉眼粗略地观察，接吻镜头多的，是爱情片。打斗镜头多的，是动作片。以我们多年的看片经验，这个分类还算合理。如果现在给我一部电影，你告诉我这个电影打斗镜头数和接吻镜头数。不告诉我这个电影类型，我可以根据你给我的信息进行判断，这个电影是属于爱情片还是动作片。而k-近邻算法也可以像我们人一样做到这一点，不同的地方在于，我们的经验更"牛逼"，而k-近邻算法是靠已有的数据。比如，你告诉我这个电影打斗镜头数为2，接吻镜头数为102，我的经验会告诉你这个是爱情片，k-近邻算法也会告诉你这个是爱情片。你又告诉我另一个电影打斗镜头数为49，接吻镜头数为51，我"邪恶"的经验可能会告诉你，这有可能是个"爱情动作片"，画面太美，我不敢想象。 (如果说，你不知道"爱情动作片"是什么？请评论留言与我联系，我需要你这样像我一样纯洁的朋友。) 但是k-近邻算法不会告诉你这些，因为在它的眼里，电影类型只有爱情片和动作片，它会提取样本集中特征最相似数据(最邻近)的分类标签，得到的结果可能是爱情片，也可能是动作片，但绝不会是"爱情动作片"。当然，这些取决于数据集的大小以及最近邻的判断标准等因素。

#### 2、距离度量

我们已经知道k-近邻算法根据特征比较，然后提取样本集中特征最相似数据(最邻近)的分类标签。那么，如何进行比较呢？比如，我们还是以表1.1为例，怎么判断红色圆点标记的电影所属的类别呢？ 如下图所示。

![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/28889705e32efe192430c93f9eec46ef.jpeg)

我们可以从散点图大致推断，这个红色圆点标记的电影可能属于动作片，因为距离已知的那两个动作片的圆点更近。k-近邻算法用什么方法进行判断呢？没错，就是距离度量。这个电影分类的例子有2个特征，也就是在2维实数向量空间，可以使用我们高中学过的两点距离公式计算距离，如图1.2所示。

![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/eb641625ed9dc104b7bad989f42ffd29.jpeg)

通过计算，我们可以得到如下结果：

-   (101,20)->动作片(108,5)的距离约为16.55
-   (101,20)->动作片(115,8)的距离约为18.44
-   (101,20)->爱情片(5,89)的距离约为118.22
-   (101,20)->爱情片(1,101)的距离约为128.69

通过计算可知，红色圆点标记的电影到动作片 (108,5)的距离最近，为16.55。如果算法直接根据这个结果，判断该红色圆点标记的电影为动作片，这个算法就是最近邻算法，而非k-近邻算法。那么k-近邻算法是什么呢？k-近邻算法步骤如下：

1.  计算已知类别数据集中的点与当前点之间的距离；
2.  按照距离递增次序排序；
3.  选取与当前点距离最小的k个点；
4.  确定前k个点所在类别的出现频率；
5.  返回前k个点所出现频率最高的类别作为当前点的预测分类。

比如，现在我这个k值取3，那么在电影例子中，按距离依次排序的三个点分别是动作片(108,5)、动作片(115,8)、爱情片(5,89)。在这三个点中，动作片出现的频率为三分之二，爱情片出现的频率为三分之一，所以该红色圆点标记的电影为动作片。这个判别过程就是k-近邻算法。

其他距离公式：  
  曼哈顿距离（ManhattanDistance）：设平面空间内存在两点，它们的坐标为(x1,y1)(x1,y1)，(x2,y2)(x2,y2)  
                          则 dis=|x1−x2|+|y1−y2|  
  ![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/003770fda926dcfb274e1553ef44b818.png)  
  比如 每个小正方形距离是1  红，栏，黄色都是12个方格都是曼哈顿距离  
           绿色线是欧氏距离(欧几里德距离：在二维和三维空间中的欧氏距离的就是两点之间的直线距离）  
  切比雪夫距离（Chebyshev Distance ）：设平面空间内存在两点，它们的坐标为(x1,y1)(x1,y1)，(x2,y2)(x2,y2)  
                          则dis=max(|x1−x2|,|y1−y2|)

  闵可夫斯基距离(MinkowskiDistance)：

  ![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/f60fa0fee2b1813e80b404022b9732fd.png)  
  两个n维变量a(x11,x12,…,x1n)与 b(x21,x22,…,x2n)间的闵可夫斯基距离定义为：   
  其中p是一个变参数。

-   当p=1时，就是曼哈顿距离
-   当p=2时，就是欧氏距离
-   当p→∞时，就是切比雪夫距离       

根据变参数的不同，闵氏距离可以表示一类的距离。 

其他距离公式参考[https://my.oschina.net/hunglish/blog/787596](https://my.oschina.net/hunglish/blog/787596)

#### 3、Python3代码实现

我们已经知道了k-近邻算法的原理，那么接下来就是使用Python3实现该算法，依然以电影分类为例。

**(1)准备数据集**

对于表1.1中的数据，我们可以使用numpy直接创建，代码如下：

```python
import numpy as np;
import matplotlib.pyplot as mp;
import collections as c;
#实现knn算法 一般用于推测 不具备学习能力 主要是比较
'
数据集合，也就是训练样本集。这个数据集有两个特征，
即打斗镜头数和接吻镜头数。
除此之外，我们也知道每个电影的所属类型，即分类标签
电影名称  打斗镜头 接吻镜头 电影类型
神雕侠侣  100       20      动作片
毒液：致命守护者  99 10     动作片
碟中谍6：全面瓦解 67  5     动作片
热情如火  40       125     动作片
泰坦尼克号    0      10     爱情片
倩女幽魂    10       20     爱情片
大话西游之月光宝盒 10  40    爱情片
烈火如歌         1    30     爱情片
'
arr=np.array([[100,200],[99,10],[67,5],[40,125],[0,10],[10,20],[10,40],[1,30]]);
tarr=np.array([1,1,1,1,0,0,0,1]);


```

**(2)k-近邻算法**

根据两点距离公式，计算距离，选择距离最小的前k个点，并返回分类结果。

```python
import numpy as np;
import matplotlib.pyplot as mp;
import collections as c;
#实现knn算法 一般用于推测 不具备学习能力 主要是比较
"
数据集合，也就是训练样本集。这个数据集有两个特征，
即打斗镜头数和接吻镜头数。
除此之外，我们也知道每个电影的所属类型，即分类标签
电影名称  打斗镜头 接吻镜头 电影类型
神雕侠侣  100       20      动作片
毒液：致命守护者  99 10     动作片
碟中谍6：全面瓦解 67  5     动作片
热情如火  40       125     动作片
泰坦尼克号    0      10     爱情片
倩女幽魂    10       20     爱情片
大话西游之月光宝盒 10  40    爱情片
烈火如歌         1    30     爱情片
"
arr=np.array([[100,200],[99,10],[67,5],[40,125],[0,10],[10,20],[10,40],[1,30]]);
tarr=np.array([1,1,1,1,0,0,0,1]);
x=arr[:,:1].T[0]
y=arr[:,1:].T[0]
print("x轴数据:",x)
print("y轴数据:",y)
#设置字体
mp.rcParams['font.family']=['STFangsong']
mp.title("电影类型图")
mp.xlabel("打斗镜头")
mp.ylabel("接吻镜头")
#第三个参数 o表示使用 散点  r表示red红色
mp.plot(x,y,"or")
mp.show();
#判断打斗镜头44  接吻镜头 12到底是哪种类型的片片了
ndata=[44,12]
#计算当前这个ndata的坐标和之前所有数据的坐标的距离 放在一个jl数组中
#距离计算公式是 欧氏距离  (x-x1)**2 +(y-y1)**2 开平方根
# jl中每个下标的数据 就是ndata和对应位置xy坐标的距离
jl=[np.sqrt((ndata[0]-i[0])**2+(ndata[0]-i[1])**2) for i in arr];
print("未排序的数据是",jl);
#对距离进行排序  然后获取排序后的下标
#  比如数组：      [10,12,8]
#  argsort升序   [2,0,1]
jlsort=np.argsort(jl);
print("排序的索引是",jlsort);
k=3;
print(jlsort[:k])
#获取指定k 前三个值最小下标的标签 也就是前三个距离最近的都是什么类型的电影
# 比如[1,1,0]
flaga=[tarr[t] for t in jlsort[:k]];
print(flaga)
#统计类型集合的哪个出现的次数 会得到一个字典
#[(1,2),(0,1)]
group=c.Counter(flaga);
#获取到个数排序（从大到小） 值最大的前1个
#[(1,2)]  [0][0]获取到1 类型就是动作片罗
print(group.most_common(1)[0][0]);
#来个三目判断下 输出中文
result=("动作片" if group.most_common(1)[0][0]==1 else "爱情片");
print(result);

```


运行结果:
```
    排序的索引是 [6 5 7 2 4 1 3 0][6 5 7][0, 0, 1]0爱情片
```


可以看到，分类结果根据我们的"经验"，是正确的，尽管这种分类比较耗时，用时1.4s。

到这里，也许有人早已经发现，电影例子中的特征是2维的，这样的距离度量可以用两 点距离公式计算，但是如果是更高维的呢？对，没错。我们可以用欧氏距离(也称欧几里德度量)，如图1.5所示。我们高中所学的两点距离公式就是欧氏距离在二维空间上的公式，也就是欧氏距离的n的值为2的情况。

![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/ee6dfe45e75caead3fea5ad77fac94a8.jpeg)

图1.5 欧氏距离公式

看到这里，有人可能会问：“分类器何种情况下会出错？”或者“答案是否总是正确的？”答案是否定的，分类器并不会得到百分百正确的结果，我们可以使用多种方法检测分类器的正确率。此外分类器的性能也会受到多种因素的影响，如分类器设置和数据集等。不同的算法在不同数据集上的表现可能完全不同。为了测试分类器的效果，我们可以使用已知答案的数据，当然答案不能告诉分类器，检验分类器给出的结果是否符合预期结果。通过大量的测试数据，我们可以得到分类器的错误率-分类器给出错误结果的次数除以测试执行的总数。错误率是常用的评估方法，主要用于评估分类器在某个数据集上的执行效果。完美分类器的错误率为0，最差分类器的错误率是1.0。同时，我们也不难发现，k-近邻算法没有进行数据的训练，直接使用未知的数据与已知的数据进行比较，得到结果。因此，可以说k-近邻算法不具有显式的学习过程。

### 二、k-近邻算法实战之约会网站配对效果判定

上一小结学习了简单的k-近邻算法的实现方法，但是这并不是完整的k-近邻算法流程，k-近邻算法的一般流程：

1.  收集数据：可以使用爬虫进行数据的收集，也可以使用第三方提供的免费或收费的数据。一般来讲，数据放在txt文本文件中，按照一定的格式进行存储，便于解析及处理。
2.  准备数据：使用Python解析、预处理数据。
3.  分析数据：可以使用很多方法对数据进行分析，例如使用Matplotlib将数据可视化。
4.  测试算法：计算错误率。
5.  使用算法：错误率在可接受范围内，就可以运行k-近邻算法进行分类。

已经了解了k-近邻算法的一般流程，下面开始进入实战内容。

#### 1、实战背景

海伦女士一直使用在线约会网站寻找适合自己的约会对象。尽管约会网站会推荐不同的任选，但她并不是喜欢每一个人。经过一番总结，她发现自己交往过的人可以进行如下分类：

1.  不喜欢的人
2.  魅力一般的人
3.  极具魅力的人

海伦收集约会数据已经有了一段时间，她把这些数据存放在文本文件datingTestSet.txt中，每个样本数据占据一行，总共有1000行。datingTestSet.txt数据下载： [约会数据](https://github.com/jiaozi789/machinelearn/blob/master/sklearn_knn/lovedata.txt)

海伦收集的样本数据主要包含以下3种特征：

1.  每年获得的飞行常客里程数
2.  玩视频游戏所消耗时间百分比
3.  每周消费的冰淇淋公升数

这里不得不吐槽一句，海伦是个小吃货啊，冰淇淋公斤数都影响自己择偶标准。打开txt文本文件，数据格式如图2.1所示。

![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/03b0ccbfdb101cbd371c8f3f462e899c.jpeg)

图2.1 datingTestSet.txt格式

#### 2、准备数据：数据解析

在将上述特征数据输入到分类器前，必须将待处理的数据的格式改变为分类器可以接收的格式。分类器接收的数据是什么格式的？从上小结已经知道，要将数据分类两部分，即特征矩阵和对应的分类标签向量。创建lovesimple.py文件，创建名为dataSet的函数，以此来处理输入格式问题。 将lovedata.txt放到与py文件相同目录下，编写代码如下：

```python
import numpy as np;
import matplotlib.pyplot as pl
import matplotlib.lines as mlines
import collections as coll;
"""
  读取data.txt的所有数据集
  前三列：
    海伦收集的样本数据主要包含以下3种特征：
        每年获得的飞行常客里程数
        玩视频游戏所消耗时间百分比
        每周消费的冰淇淋公升数
  最后一列：
    didntLike 不喜欢的人
    smallDoses 魅力一般的人
    largeDoses 极具魅力的人
  
"""
def dataSet():
    arr=[];
    with open("lovedata.txt","r") as file:
        for line in file:
            arr.append(line.strip().split("\t"));#默认删除空白符(包括'\n','\r','\t',' ')
    arrnp=np.array(arr);
    #前三列是特征数据 读取出来时字符串数组， 转换成float类型 #最后一列是标签数据 转换成1维向量
    return arrnp[:,:3].astype(dtype=np.float32),arrnp[:,3:].T[0];
``` 

运行上述代码，得到的数据解析结果如图2.2所示。

![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/4761701054d59698ef9fe4542d1c5d52.png)

可以看到，我们已经顺利导入数据，并对数据进行解析，格式化为分类器需要的数据格式。接着我们需要了解数据的真正含义。可以通过友好、直观的图形化的方式观察数据。

#### 3、分析数据：数据可视化

在lovesimple.py文件中编写名为graphDataSet的函数，用来将数据可视化。编写代码如下：

```python
def dataSet():
    arr=[];
    with open("lovedata.txt","r") as file:
        for line in file:
            arr.append(line.strip().split("\t"));#默认删除空白符(包括'\n','\r','\t',' ')
    arrnp=np.array(arr);
    #将特征数据字符串转换成数字
    return arrnp[:,:3].astype(dtype=np.float32),arrnp[:,3:].T[0];

'''
根据数据
绘制图形
'''
def graphDataSet(feature, result):
    pl.rcParams['font.family'] = ['STFangsong']
    #nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域 figsize表示画布大小
    fig,axs=pl.subplots(nrows=2,ncols=2);# """,figsize=(13,8)"""
    colorArray = ["black" if e == "didntLike" else ("orange" if e == "smallDoses" else "red") for e in result]
    drawSubPlot(axs, 0, 0, "每年获得的飞行常客里程数和玩视频游戏所消耗时间百分比占比"
                , "每年获得的飞行常客里程数",
                "玩视频游戏所消耗时间",
                feature[:, :1].T[0],
                feature[:, 1:2].T[0],
                colorArray
                )
    #####绘制 0,1这个subplot 上面代码用于学习
    drawSubPlot(axs,0,1,"玩视频游戏所消耗时间和每周消费的冰淇淋公升数占比"
                ,"玩视频游戏所消耗时间",
                "每周消费的冰淇淋公升数",
                feature[:, 1:2].T[0],
                feature[:, 2:3].T[0],
                colorArray
                )
    drawSubPlot(axs, 1, 0, "每年获得的飞行常客里程数和每周消费的冰淇淋公升数占比"
                , "每年获得的飞行常客里程数",
                "每周消费的冰淇淋公升数",
                feature[:, 0:1].T[0],
                feature[:, 2:3].T[0],
                colorArray
                )
    pl.show();
"""
  绘制子plot的封装
"""
def drawSubPlot(axs,x,y,title,xlabel,ylabel,xdata,ydata,colorArray):
    axs[x][y].set_title(title)
    axs[x][y].set_xlabel(xlabel)
    axs[x][y].set_ylabel(ylabel)
    axs[x][y].scatter(x=xdata, y=ydata, color=colorArray, s=2);
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=2, label='不喜欢')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=2, label='魅力一般')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=2, label='极具魅力')
    axs[x][y].legend(handles=[didntLike, smallDoses, largeDoses])
```
运行以下代码
```python
feature, result = dataSet()
print(feature)
print(result)
graphDataSet(feature,result)
```

，可以看到可视化结果如图所示。  
![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/41757aabae48c447cc8b532d1ebb3146.png)

通过数据可以很直观的发现数据的规律，比如以玩游戏所消耗时间占比与每年获得的飞行常客里程数，只考虑这二维的特征信息，给我的感觉就是海伦喜欢有生活质量的男人。为什么这么说呢？每年获得的飞行常客里程数表明，海伦喜欢能享受飞行常客奖励计划的男人，但是不能经常坐飞机，疲于奔波，满世界飞。同时，这个男人也要玩视频游戏，并且占一定时间比例。能到处飞，又能经常玩游戏的男人是什么样的男人？很显然，有生活质量，并且生活悠闲的人。我的分析，仅仅是通过可视化的数据总结的个人看法。我想，每个人的感受应该也是不尽相同。

#### 4、准备数据：数据归一化

以下给出了四组样本，如果想要计算样本3和样本4之间的距离，可以使用欧拉公式计算。

![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/ad71247437b8dc41170b269ebb06afdb.jpeg)

计算方法如下所示。

![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/5e5bb8e529cd3684dc2ec0c31fa74d37.jpeg)

从上计算公式

我们很容易发现，上面方程中数字差值最大的属性对计算结果的影响最大，也就是说，每年获取的飞行常客里程数对于计算结果的影响将远远大于表中其他两个特征-玩视频游戏所耗时间占比和每周消费冰淇淋公斤数的影响。而产生这种现象的唯一原因，仅仅是因为飞行常客里程数远大于其他特征值。但海伦认为这三种特征是同等重要的，因此作为三个等权重的特征之一，飞行常客里程数并不应该如此严重地影响到计算结果。

在处理这种不同取值范围的特征值时，我们通常采用的方法是将数值归一化，如将取值范围处理为０到１或者-１到１之间。下面的公式可以将任意取值范围的特征值转化为０到１区间内的值：

    newValue = (oldValue - min) / (max - min)

其中min和max分别是数据集中的最小特征值和最大特征值。虽然改变数值取值范围增加了分类器的复杂度，但为了得到准确结果，我们必须这样做。在lovsimple.py文件中编写名为normalizing的函数，用该函数自动将数据归一化。代码如下：

```python
def normalizing(feature):
    #graphDataSet(feature, result)
    #对所有的数据进行归一化
    #假设 feature=np.array([[1,2],[3,4],[1.3,2.3]])
    #每一列上的最小值  [1,2]
    minVal=np.min(feature,axis=0);
    #每一列上的最大值  [3,4]
    maxVal = np.max(feature,axis=0);
    # 当前数据集 -最小值 [[1,2],[3,4],[1.3,2.3]]-[1,2]是不行的 应该行和列一样
    # 第一列应该-1  第二列应该减去2
    # 模拟成数据  [[1,2],[3,4],[1.3,2.3]]-[[1,2],[1,2],[1,2]] 这样才行
    # 有几列 就有几个 [1,2]的最小值数组
    minArr=np.tile(minVal,(feature.shape[0],1))
    maxArr=np.tile(maxVal,(feature.shape[0],1))
    resultArr=(feature -minArr)/(maxArr-minArr);
    return resultArr;
```

添加测试代码：

    feature, result = dataSet()print(normalizing(feature))

运行上述代码，得到结果如图示。

    [[0.44832537 0.39805138 0.5623336 ] [0.1587326  0.34195465 0.9872441 ] [0.28542942 0.06892523 0.4744963 ] ...

从上面运行结果可以看到，我们已经顺利将数据归一化了  
其他比较常用的归一化方法： 均值方差归一化 

这种方式给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。经过处理的数据符合标准正态分布，即均值为0，标准差为1，转化函数为：  
![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/94cc5208e39946d5d7f3d6a71b62bc1c.gif)

其中![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/5abfc576cdaf37155547b12506ae04b2.gif)为所有样本数据的均值，![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/21948685f68d6b3c832fae1793c88444.gif)为所有样本数据的标准差。

#### 5、测试算法：验证分类器

机器学习算法一个很重要的工作就是评估算法的正确率，通常我们只提供已有数据的90%作为训练样本来训练分类器，而使用其余的10%数据去测试分类器，检测分类器的正确率。需要注意的是，10%的测试数据应该是随机选择的，由于海伦提供的数据并没有按照特定目的来排序，所以我们可以随意选择10%数据而不影响其随机性。

为了测试分类器效果，在lovsimple.py文件中创建函数datingClassTest，编写代码如下：

```python
"""
  该函数用于返回预测当前data的label值 也就是knn算法
   data 用于预测结果的数据 比如 [1000,1.1,0.8]
   trainData 是训练集  [[40920	8.326976	0.953952],[14488	7.153469	1.673904]]
   k表示预测数据最近的k个数据
   labelData 表示训练集的对应的label数据
"""
def knn(data,trainData,labelData,k):
    testData=np.tile(data,(trainData.shape[0],1))
    #print(testData)
    #计算距离差的平方开根
    sqdata=np.sqrt(np.sum((testData-trainData)**2,axis=1));
    #选取与当前点距离最小的k个点的下标；
    kindex=np.argsort(sqdata)[:k];
    #取出所有的该距离位置最近的结果
    resultdata=[labelData[ki] for ki in kindex]
    #print(sqdata)
    #print(kindex)
    #print(resultdata)
    #分组获取最大的那一个
    return (coll.Counter(resultdata).most_common(1)[0][0])

#knn(np.array([1000,1.1,0.8]),np.array([[40920,8.326976,0.953952],[14488,7.153469,1.673904],[35483,12.273169,1.508053]]),["不喜欢","喜欢","喜欢"],2)
"""
    将所有的数据按照ratio比例拆分
    90%数据用于训练  10%数据用于测试knn算法准确率
    ratio 表示拆分的比例  0.1表示训练集=1-0,1 测试集是0.1
    k表示knn的k
"""
def testData(ratio,k):
    feature, resultLabel = dataSet()
    feature=normalizing(feature);
    #拿到10%的数据用户测试knn算法
    #获取总行数
    rows=feature.shape[0];
    #获取%90的实际个数 必须将float转换成int类型
    ratioCount=int(rows*(1-ratio));
    trainData=feature[:ratioCount,];
    testData=feature[ratioCount:,];
    resultI=ratioCount;
    #统计正确率
    okCount=0;
    erroCount=0;
    for td in testData:
        realResult=resultLabel[resultI]
        calculateResult=knn(td,trainData,resultLabel,k)
        if realResult==calculateResult:
            okCount=okCount+1;
        else:
            erroCount=erroCount+1;
        print("真实结果:",realResult,"  预测结果:",calculateResult)
        resultI=resultI+1;
    print("正确率是:",(okCount/(okCount+erroCount)))
ratio=0.1;
k=5;
testData(ratio,k)
```
运行上述代码

    正确率是: 0.9504950495049505

算出正确率是: 0.95，这是一个想当不错的结果。我们可以改变函数testData内变量ratio和分类器k的值，检测错误率是否随着变量值的变化而增加。依赖于分类算法、数据集和程序设置，分类器的输出结果可能有很大的不同。

#### 6、使用算法：构建完整可用系统

我们可以给海伦一个小段程序，通过该程序海伦会在约会网站上找到某个人并输入他的信息。程序会给出她对男方喜欢程度的预测值。

在lovsimple.py文件中创建函数classifyPerson，代码如下：

```python
def classifyPerson():
     precentTats= float(input("每年获得的飞行常客里程数:"))
     ffMiles = float(input("玩视频游戏所耗时间百分比:"))
     iceCream = float(input("每周消费的冰激淋公升数:"))
     feature, resultLabel = dataSet()
     k = 10;
     calculateResult = knn([precentTats,ffMiles,iceCream], feature, resultLabel, k)
     print("您可能 ",calculateResult,"这个人")

classifyPerson()
```
在cmd中，运行程序，并输入数据(44000,12,0.5)，预测结果是"你可能有些喜欢这个人"，也就是这个人魅力一般。一共有三个档次：讨厌、有些喜欢、非常喜欢，对应着不喜欢的人、魅力一般的人、极具魅力的人。

预测结果

![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/f9327b66bf538fe402cb63e2177d2ab8.png)

以上例子换成sklearn实现,代码量大大减少

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split;
from sklearn.metrics import accuracy_score;
import sklearn.preprocessing as pre;
def dataSet():
    arr=[];
    with open("lovedata.txt","r") as file:
        for line in file:
            arr.append(line.strip().split("\t"));#默认删除空白符(包括'\n','\r','\t',' ')
    arrnp=np.array(arr);
    #将特征数据字符串转换成数字
    return arrnp[:,:3].astype(dtype=np.float32),arrnp[:,3:].T[0];

#调用sklearn测试k近邻


def testData(ratio,k):
    feature, result = dataSet()

    #使用sklearn的预处理进行归一化 使用均值方差归一化
    feature=pre.StandardScaler().fit(feature).transform(feature);
    #train_test_split将数据拆分了 test_size的比例 传入0.1就是10%的测试集
    # random_state 随即种子 可能随即抽取10%的测试集 如果random_state是某个固定的数 下次传入 获取的是相同的测试集
    # 如果是0或者不填 每次获取的测试集都不是相同的数据
    train_X, test_X, train_y, test_y=train_test_split(feature,result,test_size=ratio,random_state = 0)
    #创建一个k临近 传入距离最近的k个值
    nei = KNeighborsClassifier(k)
    #填充 训练数据 和 训练集结果
    nei.fit(train_X, train_y)
    #预测所有的测试集 得到预测的结果
    predict_y=nei.predict(test_X)
    #比较预测结果和实际结果 得到得分
    score=accuracy_score(test_y,predict_y)
    print(score)
ratio=0.1;
k=5;
testData(ratio,k);
```
运行获得结果：

    0.9405940594059405

  
 

### 三、k-近邻算法实战之sklearn手写数字识别

#### 1、实战背景

对于需要识别数字的图片一般都使用图形处理软件，处理成具有相同的色彩和大小：宽高是32像素x32像素。这里将采用本文格式存储图像，但是为了方便理解，我们将图片转换为文本格式，数字的文本格式如图所示。

[![机器学习实战教程（一）：K-近邻（KNN）算法（史诗级干货长文）](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/277fc2a1130e86b2a105e678d6f9aaeb.jpeg)](https://i-blog.csdnimg.cn/blog_migrate/277fc2a1130e86b2a105e678d6f9aaeb.jpeg)

与此同时，这些文本格式存储的数字的文件命名也很有特点，格式为：数字的值\_该数字的样本序号，如图3.2所示。

[![机器学习实战教程（一）：K-近邻（KNN）算法（史诗级干货长文）](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/88fcc52c2a64b8a4e97cb12559c5217b.jpeg)](https://i-blog.csdnimg.cn/blog_migrate/88fcc52c2a64b8a4e97cb12559c5217b.jpeg)

比如0\_0.txt 记事本打开 就是一个用0或者 1 拼成的 32\*32个字符的0  
![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/bd6e82ae0653c3ce349426b985820686.png)

比如 0\_1.txt 也是0和上面的0写法优点区别 这两个文件的label就是0  有两个样本  
![](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/474cc19402a1cb07122c52631f066ca9.png)

对于这样已经整理好的文本，我们可以直接使用Python处理，进行数字预测。数据集分为训练集和测试集，使用上小结的方法，自己设计k-近邻算法分类器，可以实现分类。数据集和实现代码下载地址：[数据集下载](https://github.com/jiaozi789/machinelearn/tree/master/sklearn_knn)  其中[trainingDigits](https://github.com/jiaozi789/machinelearn/tree/master/sklearn_knn/trainingDigits)是训练数据  
[testDigits](https://github.com/jiaozi789/machinelearn/tree/master/sklearn_knn/testDigits)是测试数据

这里不再讲解自己用Python写的k-邻域分类器的方法，因为这不是本小节的重点。接下来，我们将使用强大的第三方Python科学计算库Sklearn构建手写数字系统。

#### 2、sklearn简介

Scikit learn 也简称sklearn，是机器学习领域当中最知名的python模块之一。sklearn包含了很多机器学习的方式：

-   Classification 分类
-   Regression 回归
-   Clustering 非监督分类
-   Dimensionality reduction 数据降维
-   Model Selection 模型选择
-   Preprocessing 数据与处理

使用sklearn可以很方便地让我们实现一个机器学习算法。一个复杂度算法的实现，使用sklearn可能只需要调用几行API即可。所以学习sklearn，可以有效减少我们特定任务的实现周期。

#### 3、sklearn安装

在安装sklearn之前，需要安装两个库，即numpy+mkl和scipy。不要使用pip直接进行安装，因为pip3默安装的是numpy，而不是numpy+mkl。第三方库下载地址：[http://www.lfd.uci.edu/~gohlke/pythonlibs/](https://cuijiahua.com/wp-content/themes/begin/inc/go.php?url=http://link.zhihu.com/?target=http%3A//www.lfd.uci.edu/~gohlke/pythonlibs/)

找到对应python版本（python --version）的numpy+mkl和scipy，下载安装即可，如图3.3和图3.4所示。

[![机器学习实战教程（一）：K-近邻（KNN）算法（史诗级干货长文）](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/5893e170464a5fa7df1a72189b550abd.jpeg)](https://i-blog.csdnimg.cn/blog_migrate/5893e170464a5fa7df1a72189b550abd.jpeg)

图3.3 numpy+mkl

[![机器学习实战教程（一）：K-近邻（KNN）算法（史诗级干货长文）](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/d81013a2950e2d3a0ef605bd042fdd32.jpeg)](https://i-blog.csdnimg.cn/blog_migrate/d81013a2950e2d3a0ef605bd042fdd32.jpeg)

使用pip安装好这两个whl文件后，使用如下指令安装sklearn。

    pip install -U scikit-learn

#### 4、sklearn实现k-近邻算法简介

官网英文文档：[点我查看](https://cuijiahua.com/wp-content/themes/begin/inc/go.php?url=http://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

sklearn.neighbors模块实现了k-近邻算法，内容如图3.5所示。

[![机器学习实战教程（一）：K-近邻（KNN）算法（史诗级干货长文）](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/3a7f4197be24c59cd90fc0a1ec2c2e18.jpeg)](https://i-blog.csdnimg.cn/blog_migrate/3a7f4197be24c59cd90fc0a1ec2c2e18.jpeg)

图3.5 sklearn.neighbors

我们使用sklearn.neighbors.KNeighborsClassifier就可以是实现上小结，我们实现的k-近邻算法。KNeighborsClassifier函数一共有8个参数，如图3.6所示。

[![机器学习实战教程（一）：K-近邻（KNN）算法（史诗级干货长文）](/docs/images/content/programming/ai/machine_learning/algorithms/action_01_knn.md.images/ab1db1caa2a10e42bba715b25832c09f.jpeg)](https://i-blog.csdnimg.cn/blog_migrate/ab1db1caa2a10e42bba715b25832c09f.jpeg)

图3.6 KNeighborsClassifier

KNneighborsClassifier参数说明：

-   n\_neighbors：默认为5，就是k-NN的k的值，选取最近的k个点。
-   weights：默认是uniform，参数可以是uniform、distance，也可以是用户自己定义的函数。uniform是均等的权重，就说所有的邻近点的权重都是相等的。distance是不均等的权重，距离近的点比距离远的点的影响大。用户自定义的函数，接收距离的数组，返回一组维数相同的权重。
-   algorithm：快速k近邻搜索算法，默认参数为auto，可以理解为算法自己决定合适的搜索算法。除此之外，用户也可以自己指定搜索算法ball\_tree、kd\_tree、brute方法进行搜索，brute是蛮力搜索，也就是线性扫描，当训练集很大时，计算非常耗时。kd\_tree，构造kd树存储数据以便对其进行快速检索的树形数据结构，kd树也就是数据结构中的二叉树。以中值切分构造的树，每个结点是一个超矩形，在维数小于20时效率高。ball tree是为了克服kd树高纬失效而发明的，其构造过程是以质心C和半径r分割样本空间，每个节点是一个超球体。
-   leaf\_size：默认是30，这个是构造的kd树和ball树的大小。这个值的设置会影响树构建的速度和搜索速度，同样也影响着存储树所需的内存大小。需要根据问题的性质选择最优的大小。
-   metric：用于距离度量，默认度量是minkowski，也就是p=2的欧氏距离(欧几里德度量)。
-   p：距离度量公式。在上小结，我们使用欧氏距离公式进行距离度量。除此之外，还有其他的度量方法，例如曼哈顿距离。这个参数默认为2，也就是默认使用欧式距离公式进行距离度量。也可以设置为1，使用曼哈顿距离公式进行距离度量。
-   metric\_params：距离公式的其他关键参数，这个可以不管，使用默认的None即可。
-   n\_jobs：并行处理设置。默认为1，临近点搜索并行工作数。如果为-1，那么CPU的所有cores都用于并行工作。

#### 5、sklearn小试牛刀

我们知道数字图片是32x32的二进制图像，为了方便计算，我们可以将32x32的二进制图像转换为1x1024的向量。对于sklearn的KNeighborsClassifier输入可以是矩阵，不用一定转换为向量，不过为了跟自己写的k-近邻算法分类器对应上，这里也做了向量化处理。然后构建kNN分类器，利用分类器做预测。创建numtest.py文件，编写代码如下：

```python
import sklearn.neighbors as skn
import numpy as np;
import os
"""
获取训练集的数据
"""
def trainDataSet(dir):
    #获取目录下所有的文件名
    files=os.listdir(dir);
    label=[];
    tdata=[];
    #有多少个文件
    for i in range(len(files)) :
        fl=files[i];
        with open(dir+"/"+fl) as file:
            tdata.append([]);
            for line in file:
                line=line.strip()
                arr=[line[e] for e in range(len(line))];
                tdata[i].extend(np.array(arr).astype(np.int8));
        label.append(fl.split("_")[0])
    return tdata,label


def testData(k):
    #训练数据
    dir = "trainingDigits";
    tdata, label=trainDataSet(dir);
    ne=skn.KNeighborsClassifier(k)
    ne.fit(tdata,label);
    #测试数据
    dir1="testDigits";
    testdata, testlabel = trainDataSet(dir1);
    okCount=0;
    errCount=0;
    okCount=sum(ne.predict(testdata)==testlabel)
    print("正确个数:",okCount," 错误个数：",len(testdata)-okCount);
    print("正确率：",okCount/len(testdata));

testData(5)
```
运行上述代码，得到结果。

    正确个数: 934  错误个数： 12正确率： 0.9873150105708245

上述代码使用的algorithm参数是auto，更改algorithm参数为brute，使用暴力搜索，你会发现，运行时间变长了，变为10s+。更改n\_neighbors参数，你会发现，不同的值，检测精度也是不同的。自己可以尝试更改这些参数的设置，加深对其函数的理解。

### 四、总结

#### 1、kNN算法的优缺点

**优点**

-   简单好用，容易理解，精度高，理论成熟，既可以用来做分类也可以用来做回归；
-   可用于数值型数据和离散型数据；
-   训练时间复杂度为O(n)；无数据输入假定；
-   对异常值不敏感

**缺点**

-   计算复杂性高；空间复杂性高；
-   样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
-   一般数值很大的时候不用这个，计算量太大。但是单个样本又不能太少，否则容易发生误分。
-   最大的缺点是无法给出数据的内在含义。

#### 2、其他

-   关于algorithm参数kd\_tree的原理，可以查看《统计学方法 李航》书中的讲解；
-   关于距离度量的方法还有切比雪夫距离、马氏距离、巴氏距离等；
-   下篇文章将讲解决策树，欢迎各位的捧场！
-   如有问题，请留言。如有错误，还望指正，谢谢！

**PS： 如果觉得本篇本章对您有所帮助，欢迎关注、评论、赞！**

**参考资料：**

1.  本文中提到的电影类别分类、约会网站配对效果判定、手写数字识别实例和数据集，均来自于《机器学习实战》的第二章k-近邻算法。
2.  本文的理论部分，参考自《统计学习方法 李航》的第三章k近邻法以及《机器学习实战》的第二章k-近邻算法。