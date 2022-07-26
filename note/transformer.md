# 6.13 - 6.20

这周学的东西：

- 详细看了“Attention is all you need" 这篇论文，因为有很多术语不是很清楚，花了一点时间
- 简单看了pytorch实现transformer代码
- pytorch框架实现news classification project (AG_NEWS datasets)
- 李宏毅老师的深度学习课程4节
- Python人工智能20个小时玩转NLP自然语言处理课程
- 看了一点邱锡鹏老师的《神经网络和深度学习》



## paper reading ："Attention is all you need "

1. attention 机制首次被应用在NLP领域："Neural Machine Translation by Jointly Learing to Align and Translate"

#### 0. Abstract

<img src="/Users/zhuwenhong/Library/Application Support/typora-user-images/截屏2022-06-15 上午9.20.39.png" alt="截屏2022-06-15 上午9.20.39" style="zoom:33%;" />

```
Q：sequence transduction model ?

A：Any task where input sequences are transformed into output sequences. Eg: 【speech recognition】, 【text-to-speech】, 【machine translation】

want a single framwork able to handle as many kinds of sequence as possible


Q：English consitituency parsing ?

A：句法分析是对句子进行分析非常重要的部分。【成分句法分析】&【依存句法分析 dependency parsing 】 A constituency parse tree breaks a text into sub-phrases. Non-terminals in the tree are types of phrases, the terminals are the words in the sentence, and the edges are unlabeled. For a simple sentence "John sees Bill", a constituency parse would be:
```



```text
                Sentence
                     |
       +-------------+------------+
       |                          |
  Noun Phrase                Verb Phrase
       |                          |
     John                 +-------+--------+
                          |                |
                        Verb          Noun Phrase
                          |                |
                        sees              Bill
```



摘要所提出来的：

1. Attention 机制很早就被提出来了，最好的注意力机制模型也是通过集成encoder和decoder实现得到的。

2. Transformer仅仅基于注意力机制，不使用RNN & CNN。优点：【并行化】 & 【训练时间更少】

3. BLEU：Bilingual Evaluation Understudy （双语评估替补），代替人进行翻译结果的评估。
   - 这种指标的好处：计算代价小，快
   - easy to understand 
   - irrelavant to language itself 
   - 与人类评价结果高度相关



#### 1. Introduction

首先是讲了大量的研究工作被投入到recurrent language model & encoder-decoder这个结构（顺序处理，无法并行）。注意力机制可以使得模型不用考虑输入序列和输出序列的距离信息，Transformer 完全依赖于注意力机制，这样可以取去考虑输入 & 输出序列的全局依赖信息



```
Q : factorization tricks & conditional computation ? 

A：两篇文献（还没看💢
```



####  2. model architecture 

![截屏2022-06-15 下午4.30.00](/Users/zhuwenhong/Library/Application Support/typora-user-images/截屏2022-06-15 下午4.30.00.png)

```
Q：每一步的执行都是auto-regressive的？【AT & NAT 技术】

A：token 被一个一个地预测出来，预测第n个token的时候会把前n - 1个token看成输入。在sampling程序中，如果是用autoregressive的方式预测n个token，就要运行模型n次（每次得到的结果cat到input中，作为新的input）。如果不是auto-regressive的方式，运行一次模型就得到了结果。【工业界一般都是用auto-regressive技术】


Q：stacked self-attention ---> multi-head attention ?【暂且先这样理解吧】
```



💯：在编码器中，输入序列可能是一次性看到的，但是在解码器中，产生的y是一步一步生成的



<img src="/Users/zhuwenhong/Library/Application Support/typora-user-images/截屏2022-06-15 下午4.39.15.png" alt="截屏2022-06-15 下午4.39.15" style="zoom: 50%;" />

<img src="/Users/zhuwenhong/Library/Application Support/typora-user-images/截屏2022-06-19 下午1.58.08.png" alt="截屏2022-06-19 下午1.58.08" style="zoom:33%;" />

==encoder和decoder里面的两个箭头就是K和V==

Transformer 总体架构可以分为4个部分：

1. 输入部分
   1. 源文本嵌入层及其位置编码器
   2. 目标文本嵌入层及其位置编码器
2. 输出部分
   1. 线性层
   2. softmax层
3. 编码器部分
   1. 有N个编码器堆叠而成
   2. 每个编码器有两个子层连接而成
   3. 第一个子层连接结构包括一个多头自注意力子层和规范化层以及一个残差连接
   4. 第二个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接
4. 解码器部分
   1. 由N个解码器堆叠而成
   2. 每个解码器由三个子层连接而成
   3. 第一个子层连接结构包括一个多头自注意力子层和规范化层以及一个残差连接
   4. 第二个子层连接结构包括一个多头注意力子层和规范化层以及一个残差连接
   5. 第三个子层连接结构包括一个前馈全连接子层以及一个残差连接
5. 在decoder的部分，每次经过softmax的输出都是一个distribution，每个token都会有一个概率值，==最后还有一个EOS==【token】，这个token也是需要概率预测的。



```
Q：residual connection？

A：LayNorm(x + Sublayer(x))



Q：什么是 batch normalization & layer nornalization ？

A：BN是对一个batch内的数据的分布进行偏置，用来改善网络层间的ICS（存在争议），用来获得更加平滑的损失平面，获得更加稳定的梯度更新，加快收敛速度。通常使用在激活层之后（也有文献使用在激活层之前）。由于其是对每个batch进行分布调整，无法使用在RNN这样的序列结构以及小batch的训练之中。

LN是为了克服BN在小batch的情况下，无法使用的问题，它是对于单个样本在整个层的通道维度上进行平均。和BN一样属于输入归一化的一种。在BN和LN同时可用的情况下，BN的效果要优于LN，因为BN采用了更加丰富的样本分布信息。
```



![截屏2022-06-20 下午2.01.03](/Users/zhuwenhong/Desktop/截屏2022-06-20 下午2.01.03.png)



<img src="/Users/zhuwenhong/Library/Application Support/typora-user-images/截屏2022-06-16 下午1.26.39.png" alt="截屏2022-06-16 下午1.26.39" style="zoom:33%;" />



Q：为什么不选择additive attention，去选择dot-product ?

A：因为dot-product可以优化成矩阵运算，运行方便。==dK对于值不是很大的时候，除不除性能没有多大影响，但是如果很大的话，就会导致值做完softmax后向0，1这两点靠近，算梯度的时候就会很小，导致训练跑不动，所以需要除==

<img src="/Users/zhuwenhong/Library/Application Support/typora-user-images/截屏2022-06-19 下午2.21.27.png" alt="截屏2022-06-19 下午2.21.27" style="zoom:33%;" />

==K==是从encoder里面产生的，==Q==是从decoder里面产生的，==V==也是从encoder里面的产生的



Q：为什么采用多头注意力机制？

A：因为采用多头注意力机制可以允许模型在不同位置共同注意来自不同表示子空间的信息。==形象来说就是模拟卷及神经网络CNN中多输出通道的一个效果，可以看到更多的特征==

![截屏2022-06-19 下午2.51.04](/Users/zhuwenhong/Library/Application Support/typora-user-images/截屏2022-06-19 下午2.51.04.png)

```
n is the sequence length, d is the representation dimension, k is the kernel
size of convolutions and r the size of the neighborhood in restricted self-attention.
```

```
Q：为什么选择self- attention机制？

A：论文作者从三个方面进行说明：

1. Computational complexity per layer 
2. sequential operations (amout of computation that can be parallelized )
3. maximum path length : Learning long-range dependencies is a key-challenge 


Q：masked-attention ?，【为什么要使用masked attention呢】

A：因为在注意力机制当中，可以看到所有的输出，因此使用该方法可以避免看到后面的内容（自回归），因为采用的是自回归，根据前面一个的输出来预测第二个，如果不遮掩，就会导致看到后面的内容（encoder是一次性全部输入，decoder是一次一次的输入，是不能看到后面的内容）

```



Q，K，V的比喻解释

```text
假如我们有一个问题: 给出一段文本，使用一些关键词对它进行描述!

1. 为了方便统一正确答案，这道题可能预先已经给大家写出了一些关键词作为提示.其中这些给出的提示就可以看作是key
2. 而整个的文本信息就相当于是query
3. value的含义则更抽象，可以比作是你看到这段文本信息后，脑子里浮现的答案信息
这里我们又假设大家最开始都不是很聪明，第一次看到这段文本后脑子里基本上浮现的信息就只有提示这些信息，
因此key与value基本是相同的，但是随着我们对这个问题的深入理解，通过我们的思考脑子里想起来的东西原来越多，
并且能够开始对我们query也就是这段文本，提取关键信息进行表示.  

以上就是注意力作用的过程， 通过这个过程，我们最终脑子里的value发生了变化，根据提示key生成了query的关键词表示方法，也就是另外一种特征表示方法.

- 刚刚我们说到key和value一般情况下默认是相同，与query是不同的，这种是一般的注意力输入形式，
- 但有一种特殊情况，就是我们query与key和value相同，这种情况称为自注意力机制， 

使用一般注意力机制，是使用不同于给定文本的关键词表示它. 而自注意力机制,需要用给定文本自身来表达自己，也就是说你需要从给定文本中抽取关键词来表述它, 相当于对文本自身的一次特征提取.
```





 💯：transformer仅仅是一个功能更加强大的词袋模型而已，也就是说无论句子的结构怎么打乱，transformer都会得到类似的结果。













#### Training :

##### regularization：

正则化，是机器学习中对【原始损失函数】引入额外信息，用来==防止过拟合==，==提高模型泛化性能==的一类方法的统称。---> 目标函数变为原始损失函数 + 额外项。

常用的额外项一般有两种：L1正则化  & L2正则化（或者L1范数 & L2范数）；使用L1正则化的模型叫做Lasso回归，使用L2正则化的模型叫做Ridge回归（岭回归）
$$
min_w[\sum_{i=1}^{N}(w^Tx_i-y_i)^2+\lambda||w||_1] \\
min_w[\sum_{i=1}^{N}(w^Tx_i-y_i)^2+\lambda||w||_2^2]
$$

Byte-pair encoding ?

把词根提出来，这样就使得整个字典比较小，方便训练



训练策略中还有residual dropout & label smoothing 







#### Discussion

Transformer 最核心的技术是什么？是为了解决什么问题提出来的呢？

RNN有两个核心缺陷：

1. RNN机制，计算序列距离较远的特征之间的相关性非常困难
2. RNN机制，导致计算并行化效率低，对于计算资源的利用不够充分

transformer就是使用self- attention来解决这个问题的。

1. self-attetion : Scaled dot-product attention ，以及它形成的multi-head attention 。这种attention机制可以使得在encode阶段将模型的encode过程并行化，提高计算效率。
2. positional encoding ：高效地保留非序列输入的位置信息的方法，为了克服transformer抛弃RNN结构带来的失去位置信息的问题



缺点是positional embedding 真的可以吗？？会不会使得模型丧失了捕捉局部特征的能力？





## 代码实现



https://colab.research.google.com/drive/1P9dx60MOu3dExz-Wit9V64NNaM_D7lqC#scrollTo=s6uah6L3OIj7&uniqifier=2





