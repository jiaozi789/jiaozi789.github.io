---
title: "机器学习实战教程（六）：决策树"
date: 2025-09-18T16:55:17+08:00
weight: 1
# bookComments: false
# bookSearchExclude: false
---


# 决策树
决策树是什么？决策树(decision tree)是一种基本的分类与回归方法。举个通俗易懂的例子，如下图所示的流程图就是一个决策树，长方形代表判断模块(decision block)，椭圆形成代表终止模块(terminating block)，表示已经得出结论，可以终止运行。从判断模块引出的左右箭头称作为分支(branch)，它可以达到另一个判断模块或者终止模块。我们还可以这样理解，分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点(node)和有向边(directed edge)组成。结点有两种类型：内部结点(internal node)和叶结点(leaf node)。内部结点表示一个特征或属性，叶结点表示一个类。蒙圈没？？如下图所示的决策树，长方形和椭圆形都是结点。长方形的结点属于内部结点，椭圆形的结点属于叶结点，从结点引出的左右箭头就是有向边。而最上面的结点就是决策树的根结点(root node)。这样，结点说法就与模块说法对应上了，理解就好。

>本文大部分文字转载自https://cuijiahua.com/blog/2017/11/ml_2_decision_tree_1.html，代码和部分原创


![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7d85732ca56ea5979b0e6670e9a67d2d.png)
我们回到这个流程图，对，你没看错，这就是一个假想的相亲对象分类系统。它首先检测相亲对方是否有房。如果有房，则对于这个相亲对象可以考虑进一步接触。如果没有房，则观察相亲对象是否有上进心，如果没有，直接Say Goodbye，此时可以说："你人很好，但是我们不合适。"如果有，则可以把这个相亲对象列入候选名单，好听点叫候选名单，有点瑕疵地讲，那就是备胎。

不过这只是个简单的相亲对象分类系统，只是做了简单的分类。真实情况可能要复杂得多，考虑因素也可以是五花八门。脾气好吗？会做饭吗？愿意做家务吗？家里几个孩子？父母是干什么的？天啊，我不想再说下去了，想想都可怕。

我们可以把决策树看成一个if-then规则的集合，将决策树转换成if-then规则的过程是这样的：由决策树的根结点(root node)到叶结点(leaf node)的每一条路径构建一条规则；路径上内部结点的特征对应着规则的条件，而叶结点的类对应着规则的结论。决策树的路径或其对应的if-then规则集合具有一个重要的性质：互斥并且完备。这就是说，每一个实例都被一条路径或一条规则所覆盖，而且只被一条路径或一条规则所覆盖。这里所覆盖是指实例的特征与路径上的特征一致或实例满足规则的条件。

使用决策树做预测需要以下过程：

- 收集数据：可以使用任何方法。比如想构建一个相亲系统，我们可以从媒婆那里，或者通过采访相亲对象获取数据。根据他们考虑的因素和最终的选择结果，就可以得到一些供我们利用的数据了。
- 准备数据：收集完的数据，我们要进行整理，将这些所有收集的信息按照一定规则整理出来，并排版，方便我们进行后续处理。
- 分析数据：可以使用任何方法，决策树构造完成之后，我们可以检查决策树图形是否符合预期。
训练算法：这个过程也就是构造决策树，同样也可以说是决策树学习，就是构造一个决策树的数据结构。
- 测试算法：使用经验树计算错误率。当错误率达到了可接收范围，这个决策树就可以投放使用了。
使用算法：此步骤可以使用适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义。

# 决策树的构建的准备工作
使用决策树做预测的每一步骤都很重要，数据收集不到位，将会导致没有足够的特征让我们构建错误率低的决策树。数据特征充足，但是不知道用哪些特征好，将会导致无法构建出分类效果好的决策树模型。从算法方面看，决策树的构建是我们的核心内容。

决策树要如何构建呢？通常，这一过程可以概括为3个步骤：特征选择、决策树的生成和决策树的修剪。
## 特征选择
特征选择在于选取对训练数据具有分类能力的特征。这样可以提高决策树学习的效率，如果利用一个特征进行分类的结果与随机分类的结果没有很大差别，则称这个特征是没有分类能力的。经验上扔掉这样的特征对决策树学习的精度影响不大。通常特征选择的标准是信息增益(information gain)或信息增益比，为了简单，本文使用信息增益作为选择特征的标准。那么，什么是信息增益？在讲解信息增益之前，让我们看一组实例，贷款申请样本数据表。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/87944efd341c47ef5af629ff6383accd.png)
希望通过所给的训练数据学习一个贷款申请的决策树，用于对未来的贷款申请进行分类，即当新的客户提出贷款申请时，根据申请人的特征利用决策树决定是否批准贷款申请。

特征选择就是决定用哪个特征来划分特征空间。比如，我们通过上述数据表得到两个可能的决策树，分别由两个不同特征的根结点构成。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1b52c0981fdcc31897cad2131c43043d.png)
图(a)所示的根结点的特征是年龄，有3个取值，对应于不同的取值有不同的子结点。图(b)所示的根节点的特征是工作，有2个取值，对应于不同的取值有不同的子结点。两个决策树都可以从此延续下去。问题是：究竟选择哪个特征更好些？这就要求确定选择特征的准则。直观上，如果一个特征具有更好的分类能力，或者说，按照这一特征将训练数据集分割成子集，使得各个子集在当前条件下有最好的分类，那么就更应该选择这个特征。信息增益就能够很好地表示这一直观的准则。

什么是信息增益呢？在划分数据集之后信息发生的变化称为信息增益，知道如何计算信息增益，我们就可以计算每个特征值划分数据集获得的信息增益，获得信息增益最高的特征就是最好的选择。
**<font color=red>信息增益=整个数据的不确定性-某个特征条件的不确定=这个特征增强了多少确定性</font>**

那怎么确定数据的不确定性了，引出了香农熵的概念

### 香农熵
在可以评测哪个数据划分方式是最好的数据划分之前，我们必须学习如何计算信息增益。集合信息的度量方式称为香农熵或者简称为熵(entropy)，这个名字来源于信息论之父克劳德·香农。

如果看不明白什么是信息增益和熵，请不要着急，因为他们自诞生的那一天起，就注定会令世人十分费解。克劳德·香农写完信息论之后，约翰·冯·诺依曼建议使用"熵"这个术语，因为大家都不知道它是什么意思。

如果想彻底理解信息熵原理参考 [图解原理](https://github.com/lzeqian/machinelearntry/blob/master/learn_algorithm/%E6%9C%80%E5%A4%A7%E7%86%B5/%E5%9B%BE%E8%A7%A3%E6%9C%80%E5%A4%A7%E7%86%B5%E5%8E%9F%E7%90%86The%20Maximum%20Entropy%20Principle.png)：
其中推导用到的[拉格朗日乘子法](https://github.com/lzeqian/machinelearntry/blob/master/learn_algorithm/%E6%9C%80%E5%A4%A7%E7%86%B5/%E5%9B%BE%E8%A7%A3KKT%E6%9D%A1%E4%BB%B6%E5%92%8C%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E4%B9%98%E5%AD%90%E6%B3%95.png)
同时理解对数函数的特点：
1. 如果ax =N（a>0，且a≠1），那么数x叫做以a为底N的对数，记作x=logaN，读作以a为底N的对数，其中a叫做对数的底数，N叫做真数。
2. 如果Y=logaX  表示为Y个a相乘等于X
3. 底数a是0-1之间是单调递减 大于1是单调递增,
4. 如果a>1 x在0-1之间y负数  x=1 y=0  x>1时 y为正数
5. ln为一个算符，意思是求自然对数，即以e为底的对数。
e是一个常数，等于2.71828183…
lnx可以理解为ln(x)，即以e为底x的对数，也就是求e的多少次方等于x。
lnx=loge^x，   logeE=Ine=1

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5eec99bdf845877ff8e834f35f18d3fc.png)
熵定义为信息的期望值。在信息论与概率统计中，熵是表示随机变量不确定性的度量。如果待分类的事物可能划分在多个分类之中，则符号xi的信息定义为 ：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/25a5bf31e79d63d78c1dc5158b1b7694.png)
其中p(xi)是选择该分类的概率，比如一个团队10人中性别有男生（3）和女生（7）两个分类，上述式中的对数以2为底，也可以e为底(自然对数)。

```
男生熵=-log2 3/10 =  1.7369655941662063  
女生熵=-log2 7/10 = 0.5145731728297583
整个团队的熵=男生熵+女生熵 =2.2515387669959646
```

> 注意因为log2 p(X) 因为p(x)是概率所以是在0-1之间是负数 所以在前面加个-转换成+数
> 由于男生的熵明显大于女生，说明整个团队是男生的概率要低于女生。

  
通过上式，我们可以得到所有类别的信息。为了计算熵，我们需要计算所有类别所有可能值包含的信息期望值(数学期望)，通过下面的公式得到：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/53d8aa6c5772a19e9750cbb9ec725152.png)
期中n是分类的数目。熵越大，随机变量的不确定性就越大。

当熵中的概率由数据估计(特别是最大似然估计)得到时，所对应的熵称为经验熵(empirical entropy)。什么叫由数据估计？比如有10个数据，一共有两个类别，A类和B类。其中有7个数据属于A类，则该A类的概率即为十分之七。其中有3个数据属于B类，则该B类的概率即为十分之三。浅显的解释就是，这概率是我们根据数据数出来的。我们定义贷款申请样本数据表中的数据为训练数据集D，则训练数据集D的经验熵为H(D)，|D|表示其样本容量，及样本个数。设有K个类Ck, = 1,2,3,...,K,|Ck|为属于类Ck的样本个数，因此经验熵公式就可以写为 ：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1d91f6aa5b95419335df9ddd225f31e7.png)
根据此公式计算经验熵H(D)，分析贷款申请样本数据表中的数据。最终分类结果只有两类，即放贷和不放贷。根据表中的数据统计可知，在15个数据中，9个数据的结果为放贷，6个数据的结果为不放贷。所以数据集D的经验熵H(D)为：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/725d3399229683559d8742c29baa1e1e.png)
经过计算可知，数据集D的经验熵H(D)的值为0.971。

### 编写代码计算经验熵
在编写代码之前，我们先对数据集进行属性标注。

- 年龄：0代表青年，1代表中年，2代表老年；
- 有工作：0代表否，1代表是；
- 有自己的房子：0代表否，1代表是；
- 信贷情况：0代表一般，1代表好，2代表非常好；
- 类别(是否给贷款)：no代表否，yes代表是。
- 确定这些之后，我们就可以创建数据集，并计算经验熵了，代码编写如下：

```
#%%

#数据集，yes表示放贷，no表示不放贷
'''
具体参考：案例图
 特征1表示年龄 0表示青年，1表示中间，2表示老年
 特征2表示是否有工作  0表示否，1表示有
 特征3表示是否有自己的房子 0表示否 1表示有
 特征4是信贷情况 0表示一般 1表示好 2表示非常好。
'''
import numpy as np
dataSet = np.array([[0, 0, 0, 0, 'no'],         
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']])
labels = ['不放贷', '放贷']
'''
  计算经验熵
  D代表传入的数据集
'''
def ShannonEnt(D):
    #去除重复元素的结论的分类:[yes,no]
    kArray=np.unique(D[:,4].reshape(1,len(D)))
    #计算出最终分类的个数
    k=len(kArray)
    #获取整个样本集的个数
    D_count=len(D)
    #经验熵
    HD=0;
    #循环多个分类，计算这个分类的熵，最后求和
    for i in range(k):
        #获取等于当前分类的数据行
        ck=[row for row in D if row[4]==kArray[i]]
        HD-=len(ck)/D_count *np.log2(len(ck)/D_count) 
    return HD;
HD_=ShannonEnt(dataSet)
print("整个数据经验熵：",HD_)  
```
输出结果：
整个数据经验熵： 0.9709505944546686

### 条件熵
熵我们知道是什么，条件熵又是个什么鬼？条件熵H(Y|X)表示在已知随机变量X的条件下随机变量Y的不确定性，随机变量X给定的条件下随机变量Y的条件熵(conditional entropy)H(Y|X)，定义为X给定条件下Y的条件概率分布的熵对X的数学期望：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/123cf176150f1554483dbf842ad5c642.png)
设特征A有n个不同的取值{a1,a2,···,an}，根据特征A的取值将D划分为n个子集{D1,D2，···,Dn}，|Di|为Di的样本个数。记子集Di中属于Ck的样本的集合为Dik，即Dik = Di ∩ Ck，|Dik|为Dik的样本个数。于是经验条件熵的公式可以些为：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/24a97127b8860c4bdefe1dd526f4ab7b.png)
其实就是：求和（获取当前特征相同数据集在整个数据的概率 *  这个数据集中最终分类的熵）

```
'''
  计算条件熵
  H(D|0) 计算某个特征列的条件熵，当年龄特征的情况下，是否房贷不确定的情况，越大越不确定
'''     
def calcConditionShannon(D,index):
    #去除重复元素的index列的数组
    featureType=np.unique(D[:,index].reshape(1,len(D)))
    featureTypeCount=len(featureType)
    #获取整个样本集的个数
    D_count=len(D)
    HDA=0;
    for i in range(featureTypeCount):
        Di=np.array([row for row in D if row[index]==featureType[i]])
        HDA+=len(Di)/D_count*ShannonEnt(Di)
    return HDA;
print("年龄特征条件熵",calcConditionShannon(dataSet,0))
```
输出：年龄特征条件熵 0.8879430945988998

### 信息增益
**<font color=red>信息增益=整个数据的不确定性-某个特征条件的不确定=这个特征增强了多少确定性</font>**
**<font color=red>信息增益=经验熵-当前特征条件熵</font>**
信息增益是相对于特征而言的，信息增益越大，特征对最终的分类结果影响也就越大，我们就应该选择对最终分类结果影响最大的那个特征作为我们的分类特征。

明确了条件熵和经验条件熵的概念。接下来，让我们说说信息增益。前面也提到了，信息增益是相对于特征而言的。所以，特征A对训练数据集D的信息增益g(D,A)，定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H(D|A)之差，即：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e7d44ed1d6c9674a5f3d1f39515b3e98.png)
说了这么多概念性的东西，没有听懂也没有关系，举几个例子，再回来看一下概念，就懂了。

以贷款申请样本数据表为例进行说明。看下年龄这一列的数据，也就是特征A1，一共有三个类别，分别是：青年、中年和老年。我们只看年龄是青年的数据，年龄是青年的数据一共有5个，所以年龄是青年的数据在训练数据集出现的概率是十五分之五，也就是三分之一。同理，年龄是中年和老年的数据在训练数据集出现的概率也都是三分之一。现在我们只看年龄是青年的数据的最终得到贷款的概率为五分之二，因为在五个数据中，只有两个数据显示拿到了最终的贷款，同理，年龄是中年和老年的数据最终得到贷款的概率分别为五分之三、五分之四。所以计算年龄的信息增益，过程如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/93fd92d463febda08735d72e73fe5e47.png)
同理，计算其余特征的信息增益g(D,A2)、g(D,A3)和g(D,A4)。分别为：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/096649bbc87c61de08709ea1709c7ea2.png)
最后，比较特征的信息增益，由于特征A3(有自己的房子)的信息增益值最大，所以选择A3作为最优特征。

我们已经学会了通过公式计算信息增益，接下来编写代码，计算信息增益。

```
'''
    计算某个特征的信息增益
    信息增益=整个数据的不确定性-某个特征条件的不确定=这个特征增强了多少确定性
'''
def calaInfoGrain(D,index):
    return ShannonEnt(dataSet)-calcConditionShannon(D,index)
print("年龄的信息增益",HD_-calcConditionShannon(dataSet,0))
print("工作的信息增益",calaInfoGrain(dataSet,1))

feature_count=len(dataSet[0])
for i in range(feature_count-1):
    print("第"+str(i)+"个特征的信息增益",HD_-calcConditionShannon(dataSet,i))
```
输出：
年龄的信息增益 0.08300749985576883
工作的信息增益 0.32365019815155627
第0个特征的信息增益 0.08300749985576883
第1个特征的信息增益 0.32365019815155627
第2个特征的信息增益 0.4199730940219749
第3个特征的信息增益 0.36298956253708536

对比我们自己计算的结果，发现结果完全正确！最优特征的索引值为2，也就是特征A3(有自己的房子)。

## 决策树的生成
我们已经学习了从数据集构造决策树算法所需要的子功能模块，包括经验熵的计算和最优特征的选择，其工作原理如下：得到原始数据集，然后基于最好的属性值划分数据集，由于特征值可能多于两个，因此可能存在大于两个分支的数据集划分。第一次划分之后，数据集被向下传递到树的分支的下一个结点。在这个结点上，我们可以再次划分数据。因此我们可以采用递归的原则处理数据集。

构建决策树的算法有很多，比如C4.5、ID3和CART，这些算法在运行时并不总是在每次划分数据分组时都会消耗特征。由于特征数目并不是每次划分数据分组时都减少，因此这些算法在实际使用时可能引起一定的问题。目前我们并不需要考虑这个问题，只需要在算法开始运行前计算列的数目，查看算法是否使用了所有属性即可。

决策树生成算法递归地产生决策树，直到不能继续下去未为止。这样产生的树往往对训练数据的分类很准确，但对未知的测试数据的分类却没有那么准确，即出现过拟合现象。过拟合的原因在于学习时过多地考虑如何提高对训练数据的正确分类，从而构建出过于复杂的决策树。解决这个问题的办法是考虑决策树的复杂度，对已生成的决策树进行简化。
### 决策树构建
ID3算法的核心是在决策树各个结点上对应信息增益准则选择特征，递归地构建决策树。具体方法是：从根结点(root node)开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子节点；再对子结点递归地调用以上方法，构建决策树；直到所有特征的信息增益均很小或没有特征可以选择为止。最后得到一个决策树。ID3相当于用极大似然法进行概率模型的选择。
#### ID3算法
在使用ID3构造决策树之前，我们再分析下数据。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3eac4922093e62fcad3d6aa3d6dc0c39.png)
按照第三列排好序的数据

```
print(dataSet[np.argsort(dataSet[:,2])])
输出：
[['0' '0' '0' '0' 'no']
 ['0' '0' '0' '1' 'no']
 ['0' '1' '0' '1' 'yes']
 ['0' '0' '0' '0' 'no']
 ['1' '0' '0' '0' 'no']
 ['1' '0' '0' '1' 'no']
 ['2' '1' '0' '1' 'yes']
 ['2' '1' '0' '2' 'yes']
 ['2' '0' '0' '0' 'no']
 ['0' '1' '1' '0' 'yes']
 ['1' '1' '1' '1' 'yes']
 ['1' '0' '1' '2' 'yes']
 ['1' '0' '1' '2' 'yes']
 ['2' '0' '1' '2' 'yes']
 ['2' '0' '1' '1' 'yes']]
```


由于特征A3(有自己的房子)的信息增益值最大，所以选择特征A3作为根结点的特征。它将训练集D划分为两个子集D1(A3取值为"是")和D2(A3取值为"否")。由于D1只有同一类的样本点，所以它成为一个叶结点，结点的类标记为“是”。
其中D1就是
```
 ['0' '1' '1' '0' 'yes']
 ['1' '1' '1' '1' 'yes']
 ['1' '0' '1' '2' 'yes']
 ['1' '0' '1' '2' 'yes']
 ['2' '0' '1' '2' 'yes']
 ['2' '0' '1' '1' 'yes']]
```
由于D1=1的时候只有一个分类结论yes，所以他是一个叶子节点，没有分叉
D2就是
```
['0' '0' '0' '0' 'no']
 ['0' '0' '0' '1' 'no']
 ['0' '1' '0' '1' 'yes']
 ['0' '0' '0' '0' 'no']
 ['1' '0' '0' '0' 'no']
 ['1' '0' '0' '1' 'no']
 ['2' '1' '0' '1' 'yes']
 ['2' '1' '0' '2' 'yes']
 ['2' '0' '0' '0' 'no']
```

对D2则需要从特征A1(年龄)，A2(有工作)和A4(信贷情况)中选择新的特征，计算各个特征的信息增益：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/88bebaf447b1f69c9a99ec011c529f7c.png)
根据计算，选择信息增益最大的特征A2(有工作)作为结点的特征。由于A2有两个可能取值，从这一结点引出两个子结点：一个对应"是"(有工作)的子结点，包含3个样本，它们属于同一类，所以这是一个叶结点，类标记为"是"；另一个是对应"否"(无工作)的子结点，包含6个样本，它们也属于同一类，所以这也是一个叶结点，类标记为"否"。
剩余数据按照A2排序
```
[['0' '0' '0' '0' 'no']
 ['0' '0' '0' '1' 'no']
 ['0' '0' '0' '0' 'no']
 ['1' '0' '0' '0' 'no']
 ['1' '0' '0' '1' 'no']
 ['2' '0' '0' '0' 'no']
 ['0' '1' '0' '1' 'yes']
 ['2' '1' '0' '1' 'yes']
 ['2' '1' '0' '2' 'yes']]
```
发现A2=0的结果全是no，A1等于1的全部为yes，所以没有其他分类了，到工作这里节点就结束了
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/287a2e61a64e4369524d95cb725b0b90.png)
可以理解为叶子节点就是结论是否贷款，分叉就是特征的值。
#### 编写代码构建决策树
我们使用字典存储决策树的结构，比如上小节我们分析出来的决策树，用字典可以表示为：
```
{'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
```
代码实现如下

```
#%%

'''
  将数据按照值指定特征列分组，，比如有房子=1的数据行和无房子=0的数据行
  {
     0:[[]]
     1:[[]]
  }
'''
colLabels=["年龄","有工作","有自己的房子","信贷情况"]
def splitData(D,index):
    kArray=np.unique(D[:,index].reshape(1,len(D)))
    #循环多个分类，计算这个分类的熵，最后求和
    returnJSon={};
    for i in range(len(kArray)):
        #获取等于当前分类的数据行
        ck=[row for row in D if row[index]==kArray[i]]
        returnJSon[i]=np.array(ck)
    return returnJSon;
    
def createDecisionTree(D):
    buildTree=None
    #如果传入的D没有数据或者第5列（是否贷款）只有一个分类值，就说明已经是叶子节点了,直接返回结论值
    resultUniqueArray=np.unique(D[:,4].reshape(1,len(D)))
    print(resultUniqueArray,len(D),len(resultUniqueArray))
    if(len(D)==0 or len(resultUniqueArray)==1):
        return resultUniqueArray[0]
    #获取特征数
    feature_count=D.shape[1]
    #算出每个特征的信息增益
    grain=[calaInfoGrain(D,i)for i in range(feature_count-1)]
    #获取信息增益最大的特征值
    maxFeatureIndex=np.argmax(grain);
    #创建一个json对象，里面有个当前特征名称的对象:比如{'有自己的房子': {}}
    buildTree={colLabels[maxFeatureIndex]:{}};
    #循环每个独立的特征值 
    featureGroup=splitData(D,maxFeatureIndex)
    for featureValue in featureGroup:
        buildTree[colLabels[maxFeatureIndex]][featureValue]=createDecisionTree(featureGroup[featureValue])
    return buildTree;
        
    
print(createDecisionTree(dataSet))
```
### 决策树可视化
以内graphviz简单易懂，这里使用graphviz来进行可视化
下载[graphviz](http://www.graphviz.org/download/)，选择将环境加入到PATH中。
python安装组件
```
pip install graphviz
```
代码绘制

```
from graphviz import Digraph
import uuid

def graphDecisionTree(dot,treeNode,parentName,lineName):
    for key in treeNode:
        if type(key)==int:
            if type(treeNode[key])==str or type(treeNode[key])==np.str_:
                #因为会出现两个yes，所以可能不能出现一个分叉而直接指向了，所以名字加上个uuid区分
                node_name=str(treeNode[key])+str(uuid.uuid1())
                dot.node(name=node_name, label=str(treeNode[key]), color='red',fontname="Microsoft YaHei")
                dot.edge(str(parentName),str(node_name), label=str(key), color='red')
            else:
                graphDecisionTree(dot,treeNode[key],parentName,key)
        elif type(treeNode[key])==dict:
            graphDecisionTree(dot,treeNode[key],key,None)
        if type(key)==str or type(treeNode[key])==str:
            dot.node(name=key, label=str(key), color='red',fontname="Microsoft YaHei")
        if parentName is not None and lineName is not None:
            dot.edge(parentName,key, label=str(lineName), color='red')
dot = Digraph(name="pic", comment="测试", format="png")
graphDecisionTree(dot,decisionTreeJson,None,None)
dot.render(filename='my_pic',
               directory='.',  # 当前目录
               view=True)
```
输出流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5352c309eec2e5d27c4bbe5746e581ff.png)
### 使用决策树执行分类
依靠训练数据构造了决策树之后，我们可以将它用于实际数据的分类。在执行数据分类时，需要决策树以及用于构造树的标签向量。然后，程序比较测试数据与决策树上的数值，递归执行该过程直到进入叶子结点；最后将测试数据定义为叶子结点所属的类型。在构建决策树的代码，可以看到，有个featLabels参数。它是用来干什么的？它就是用来记录各个分类结点的，在用决策树做预测的时候，我们按顺序输入需要的分类结点的属性值即可。举个例子，比如我用上述已经训练好的决策树做分类，那么我只需要提供这个人是否有房子，是否有工作这两个信息即可，无需提供冗余的信息。

用决策树做分类的代码很简单，编写代码如下：

```
#%%
'''
    在决策树中判断传入的特征是否贷款
'''
def classfiy(decisionTreeJson,featureLabel,vecTest,index):
    if type(decisionTreeJson)==str or type(decisionTreeJson)==np.str_:
        return decisionTreeJson
    elif type(decisionTreeJson[featureLabel[index]])==dict :
        return classfiy(decisionTreeJson[featureLabel[index]][vecTest[index]],featureLabel,vecTest,index+1)
    else :
        return decisionTreeJson

print("是" if classfiy(decisionTreeJson,featureLabel,[1,0],0)=='yes' else "否")
```
### 决策树的存储
使用序列化的方式即可

```
  import pickle
  #写入
  with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
#读取
 fr = open(filename, 'rb')
    json=pickle.load(fr)
```
### Sklearn之使用决策树预测隐形眼睛类型
#### 实战背景
进入本文的正题：眼科医生是如何判断患者需要佩戴隐形眼镜的类型的？一旦理解了决策树的工作原理，我们甚至也可以帮助人们判断需要佩戴的镜片类型。

隐形眼镜数据集是非常著名的数据集，它包含很多换着眼部状态的观察条件以及医生推荐的隐形眼镜类型。隐形眼镜类型包括硬材质(hard)、软材质(soft)以及不适合佩戴隐形眼镜(no lenses)。数据来源与UCI数据库，数据集下载地址：https://github.com/lzeqian/machinelearntry/blob/master/sklearn_decisiontree/lenses.txt

一共有24组数据，数据的Labels依次是age、prescript、astigmatic、tearRate、class，也就是第一列是年龄，第二列是症状，第三列是是否散光，第四列是眼泪数量，第五列是最终的分类标签。数据如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7c70073999f8a5c565d421d72e150143.png)
可以使用已经写好的Python程序构建决策树，不过出于继续学习的目的，本文使用Sklearn实现。

#### 使用Sklearn构建决策树
官方英文文档地址：http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

sklearn.tree模块提供了决策树模型，用于解决分类问题和回归问题。方法如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/387a569c217158aeeceeff4e7ea54428.png)
本次实战内容使用的是DecisionTreeClassifier和export_graphviz，前者用于决策树构建，后者用于决策树可视化。

DecisionTreeClassifier构建决策树：

让我们先看下DecisionTreeClassifier这个函数，一共有12个参数：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b4243b3c381604d8f0d98e3b2260503f.png)

参数说明如下：

- criterion：特征选择标准，可选参数，默认是gini，可以设置为entropy。gini是基尼不纯度，是将来自集合的某种结果随机应用于某一数据项的预期误差率，是一种基于统计的思想。entropy是香农熵，也就是上篇文章讲过的内容，是一种基于信息论的思想。Sklearn把gini设为默认参数，应该也是做了相应的斟酌的，精度也许更高些？ID3算法使用的是entropy，CART算法使用的则是gini。
- splitter：特征划分点选择标准，可选参数，默认是best，可以设置为random。每个结点的选择策略。best参数是根据算法选择最佳的切分特征，例如gini、entropy。random随机的在部分划分点中找局部最优的划分点。默认的"best"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"random"。
-max_features：划分时考虑的最大特征数，可选参数，默认是None。寻找最佳切分时考虑的最大特征数(n_features为总共的特征数)，有如下6种情况：
1. 如果max_features是整型的数，则考虑max_features个特征；
2. 如果max_features是浮点型的数，则考虑int(max_features * n_features)个特征；
3. 如果max_features设为auto，那么max_features = sqrt(n_features)；
4. 如果max_features设为sqrt，那么max_featrues = sqrt(n_features)，跟auto一样；
5. 如果max_features设为log2，那么max_features = log2(n_features)；
6. 如果max_features设为None，那么max_features = n_features，也就是所有特征都用。
7. 一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。
- max_depth：决策树最大深，可选参数，默认是None。这个参数是这是树的层数的。层数的概念就是，比如在贷款的例子中，决策树的层数是2层。如果这个参数设置为None，那么决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。或者如果设置了min_samples_slipt参数，那么直到少于min_smaples_split个样本为止。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。
- min_samples_split：内部节点再划分所需最小样本数，可选参数，默认是2。这个值限制了子树继续划分的条件。如果min_samples_split为整数，那么在切分内部结点的时候，min_samples_split作为最小的样本数，也就是说，如果样本已经少于min_samples_split个样本，则停止继续切分。如果min_samples_split为浮点数，那么min_samples_split就是一个百分比，ceil(min_samples_split * n_samples)，数是向上取整的。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
- min_samples_leaf：叶子节点最少样本数，可选参数，默认是1。这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。叶结点需要最少的样本数，也就是最后到叶结点，需要多少个样本才能算一个叶结点。如果设置为1，哪怕这个类别只有1个样本，决策树也会构建出来。如果min_samples_leaf是整数，那么min_samples_leaf作为最小的样本数。如果是浮点数，那么min_samples_leaf就是一个百分比，同上，celi(min_samples_leaf * n_samples)，数是向上取整的。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
- min_weight_fraction_leaf：叶子节点最小的样本权重和，可选参数，默认是0。这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。
- max_leaf_nodes：最大叶子节点数，可选参数，默认是None。通过限制最大叶子节点数，可以防止过拟合。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
- class_weight：类别权重，可选参数，默认是None，也可以字典、字典列表、balanced。指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。类别的权重可以通过{class_label：weight}这样的格式给出，这里可以自己指定各个样本的权重，或者用balanced，如果使用balanced，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个参数，选择默认的None。
- random_state：可选参数，默认是None。随机数种子。如果是证书，那么random_state会作为随机数生成器的随机数种子。随机数种子，如果没有设置随机数，随机出来的数与当前系统时间有关，每个时刻都是不同的。如果设置了随机数种子，那么相同随机数种子，不同时刻产生的随机数也是相同的。如果是RandomState instance，那么random_state是随机数生成器。如果为None，则随机数生成器使用np.random。
- min_impurity_split：节点划分最小不纯度,可选参数，默认是1e-7。这是个阈值，这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。
presort：数据是否预排序，可选参数，默认为False，这个值是布尔值，默认是False不排序。一般来说，如果样本量少或者限制了一个深度很小的决策树，设置为true可以让划分点选择更加快，决策树建立的更加快。如果样本量太大的话，反而没有什么好处。问题是样本量少的时候，我速度本来就不慢。所以这个值一般懒得理它就可以了。

除了这些参数要注意以外，其他在调参时的注意点有：

- 当样本数量少但是样本特征非常多的时候，决策树很容易过拟合，一般来说，样本数比特征数多一些会比较容易建立健壮的模型
- 如果样本数量少但是样本特征非常多，在拟合决策树模型前，推荐先做维度规约，比如主成分分析（PCA），特征选择（Losso）或者独立成分分析（ICA）。这样特征的维度会大大减小。再来拟合决策树模型效果会好。
- 推荐多用决策树的可视化，同时先限制决策树的深度，这样可以先观察下生成的决策树里数据的初步拟合情况，然后再决定是否要增加深度。
- 在训练模型时，注意观察样本的类别情况（主要指分类树），如果类别分布非常不均匀，就要考虑用class_weight来限制模型过于偏向样本多的类别。
- 决策树的数组使用的是numpy的float32类型，如果训练数据不是这样的格式，算法会先做copy再运行。
- 如果输入的样本矩阵是稀疏的，推荐在拟合前调用csc_matrix稀疏化，在预测前调用csr_matrix稀疏化。

sklearn.tree.DecisionTreeClassifier()提供了一些方法供我们使用，如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1f96143acc4c04c5bd9610ca67547a59.png)
了解到这些，我们就可以编写代码了。
注意： 因为在fit()函数不能接收string类型的数据，通过打印的信息可以看到，数据都是string类型的。在使用fit()函数之前，我们需要对数据集进行编码，这里可以使用两种方法：

- LabelEncoder ：将字符串转换为增量值
- OneHotEncoder：使用One-of-K算法将字符串转换为整数

为了对string类型的数据序列化，需要先生成pandas数据，这样方便我们的序列化工作。这里我使用的方法是，原始数据->字典->pandas数据，编写代码如下：

```
#%%

import numpy as np
import pandas as pd
fr = open('lenses.txt')
lenses = np.array([inst.strip().split('\t') for inst in fr.readlines()])
print(lenses)
#四个特征一列是：年龄，第二列是症状，第三列是是否散光，第四列是眼泪数量
#，第五列是最终的分类标签，隐形眼镜类型包括硬材质(hard)、软材质(soft)以及不适合佩戴隐形眼镜(no lenses)
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
#最终分类在最后一列
lenses_target = [each[-1] for each in lenses]                                                        
print(lenses_target)

#%%

#组装成带有表头的数据格式
lensesDataFrame=np.concatenate((np.array([lensesLabels]),lenses[:,0:4]))
'''
注意dataframe的用法
df['a']#取a列
df[['a','b']]#取a、b列
默认的表头是0，1，2这样的序号，如果需要自定义表头需要定义json
{
   "age":[young,pre],
   "prescript":["myope","myope"]
}
'''
jsonData= {l:lenses[:,i]for i,l in enumerate(lensesLabels)}
lenses_pd = pd.DataFrame(jsonData)                                    #生成pandas.DataFrame
print(lenses_pd)


#%%

from sklearn.preprocessing import LabelEncoder
# 将所有的label 比如young转换成0，pre转为成1这样的数字编码
le = LabelEncoder()          
#传入一个一维的数字，在这个数组里，相同的字符串转换为相同的数字
for i in lenses_pd.columns:
    lenses_pd[i]=le.fit_transform(lenses_pd[i])
print(lenses_pd)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5b061a57e97b4d1c7594bf93ef9b7875.png)

#### 使用Graphviz可视化决策树
graphviz之前已经安装过了，安装一个pydotplus库

```
pip3 install pydotplus
```
编写代码

```

#使用sklearn决策树
from sklearn import tree
import pydotplus
from io import StringIO
clf = tree.DecisionTreeClassifier(max_depth = 4)                        #创建DecisionTreeClassifier()类
clf = clf.fit(lenses_pd.values.tolist(), lenses_target)                    #使用数据，构建决策树
dot_data = StringIO()
tree.export_graphviz(clf, out_file = dot_data,                            #绘制决策树
                    feature_names = lenses_pd.keys(),
                    class_names = clf.classes_,
                    filled=True, rounded=True,
                    special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("tree.pdf")     
```

运行代码，在该python文件保存的相同目录下，会生成一个名为tree的PDF文件，打开文件，我们就可以看到决策树的可视化效果图了。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/828883577aeae87df5ee7bb18c745db5.png)
确定好决策树之后，我们就可以做预测了。可以根据自己的眼睛情况和年龄等特征，看一看自己适合何种材质的隐形眼镜。使用如下代码就可以看到预测结果：

```
print(clf.predict([[1,1,1,0]]))   
```
## 总结
决策树的一些优点：

- 易于理解和解释。决策树可以可视化。
- 几乎不需要数据预处理。其他方法经常需要数据标准化，创建虚拟变量和删除缺失值。决策树还不支持缺失值。
- 使用树的花费（例如预测数据）是训练数据点(data points)数量的对数。
- 可以同时处理数值变量和分类变量。其他方法大都适用于分析一种变量的集合。
- 可以处理多值输出变量问题。
- 使用白盒模型。如果一个情况被观察到，使用逻辑判断容易表示这种规则。相反，如果是黑盒模型（例如人工神经网络），结果会非常难解释。
- 即使对真实模型来说，假设无效的情况下，也可以较好的适用。


决策树的一些缺点：

- 决策树学习可能创建一个过于复杂的树，并不能很好的预测数据。也就是过拟合。修剪机制（现在不支持），设置一个叶子节点需要的最小样本数量，或者数的最大深度，可以避免过拟合。
- 决策树可能是不稳定的，因为即使非常小的变异，可能会产生一颗完全不同的树。这个问题通过decision trees with an ensemble来缓解。
- 概念难以学习，因为决策树没有很好的解释他们，例如，XOR, parity or multiplexer problems。
- 如果某些分类占优势，决策树将会创建一棵有偏差的树。因此，建议在训练之前，先抽样使样本均衡。
其他：