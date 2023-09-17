# Residual Log-likelihood Estimation（分布与回归）

## 1.数学分布

### **回归**的思想

在往常的自上至下的回归中，可以看出通常使用的都是L2或L1损失，这是自然而然的想法，也是这些分布在自然界中很常见也有着良好性质。

直观上来说，根据完美的数据回归出的坐标值是一个定值，为什么以分布来表示呢。其实对数据而言，必定有一个噪声$\epsilon$, 可以说噪声的分布影响了数据的分布，也影响着误差函数的选择。例如用 $y = w^T x$ 来预测y, 而已知y被高斯噪声$\epsilon$所影响。故可以写为$y = w^T x + \epsilon$  ,又 $\epsilon \sim (0, \sigma )$ 所以可以得出，$y \sim (w^T x, \sigma)$由此可以说 $y$ 服从均值为 $w^x$ 方差为 $\epsilon$ 的高斯分布（一切都基于噪声为高斯分布的假设上而得出）。

而为什么又说L2损失是最适合此分布的呢？由最大似然函数可得出。对于上述 $y$ 而言，其分布的数学表达形式为 :
$$
P(y) = \frac{1}{\sqrt{2\pi} \sigma } e^{(-\frac{(y-w^T x)^2}{2\sigma ^2} )}
$$
根据极大似然估计得出：
$$
w_{MLE} = \underset{w}{argmax}log(P(y)) =  \underset{w}{argmin}\frac{1}{N} \sum_{i=1}^{N} (\frac{(y-w^T x)^2}{2\sigma ^2} )
$$
对可变量调整使上式最小，即满足最大分布估计。故误差函数为:
$$
Loss(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y-w^T x)^2
$$
证明了回归预测结果的分布是与噪声分布息息相关的，使网络回归出具体坐标点相当于预测了分布的 $\mu$ , 而在RLE中还同时预测了 $\sigma$ 并使得分布不再仅先验地认为使高斯或拉普拉斯分布，而是通过流模型向真实分布靠近。

## 2.论文思想

首先，即使自然界中拉普拉斯或高斯分布都具有十分优良的性质，且对于数据的拟合效果优秀。但不可否认，真实分布永远和先验分布有着不小的差距，尤其是在训练数据出现变化，数据集种类不同的时候，所以本文采用流模型来学习真实的分布。流可以理解为，通过一系列可逆的变换函数，例如多次重复简单算式$y = w^T x + b$ 就能使一个如高斯分布等简单分布，变化为一个复杂分布。

对于以往的传统的回归网络，通常目标是拟合一个高斯或拉普拉斯分布。在此基础上进一步将此简单分布以流模型变换为真实分布以达到更好的效果。

### 	                        

​    

### basic design

这是最为朴素的一种思想，因为已知我们通过L2损失函数实则学习到的是目标关键点的高斯分布，而后我们对此分布进行流模型变换，就能将一个简单分布拟合为一个复杂分布了，而在实际中此法是不可行的。因为首先已经学习到了一个简单分布了，对此进行流模型变换是否在实际中就是增加FC层呢？如果整体进行训练，此FC层可能并无作用。如若分开训练，则会使步骤繁琐。

<img src="https://s2.loli.net/2023/09/11/Z62mV5EforLbiwO.png" alt="image-20230911142653307" style="zoom: 80%;" />

用极大似然方法（Maximum Likelihood Estimation)，可以直观地推断出LOSS函数：


$$
\begin{aligned}
\mathcal{L}_{m l e} & =-\left.\log P_{\Theta, \phi}(\mathbf{x} \mid \mathcal{I})\right|_{\mathbf{x}=\boldsymbol{\mu}_{g}} \\
& =-\log P_{\Theta}\left(f_{\phi}^{-1}\left(\boldsymbol{\mu}_{g}\right) \mid \mathcal{I}\right)-\log \left|\operatorname{det} \frac{\partial f_{\phi}^{-1}}{\partial \boldsymbol{\mu}_{g}}\right|
\end{aligned}
$$
因为在Basic Design中的思想就是对已然预测到的简单分布进行变换 ，设预测到的分布为$P_{\Theta}\left(z \mid \mathcal{I}\right)$ ,进行反向流模型变换$f_{\phi}^{-1}\left(\boldsymbol{x}\right)$后即可以将$x$带入原分布中，即可完成Loss函数的推导。

### Reparameterization

<img src="https://s2.loli.net/2023/09/11/fkWFDInOEa8mAUY.png" alt="image-20230911145942534" style="zoom:80%;" />

在实际工程实践中，流模型与回归模型的并行学习能取得更好的效果，同时也避免了two-stage的工作流程。我们以先验知识获取一个简单的标准化分布，对其进行变换，让网络直接学习真实分布更为合理。我们以标准正态分布为基础分布，通过流模型将基础分布变换为真实分布的原始分布，用回归网络来预测平移和缩放系数 $\mu$ 和 $\sigma$ 。这里的平移和缩放参数加上预测得到的标准化真实分布，就能得到真是关键点分布。
$$
\begin{aligned}
\mathcal{L}_{m l e} & =-\left.\log P_{\Theta, \phi}(x \mid \mathcal{I})\right|_{\mathbf{x}=\boldsymbol{\mu}_{g}} \\
& =-\log P_{\Theta}\left(\bar{\mu} _{g} \mid \mathcal{I}\right)-\log \left|\operatorname{det} \frac{\partial \bar{\mu} _{g}}{\partial \boldsymbol{\mu}_{g}}\right|
\end{aligned}
$$
其中推导过程理解为，有一个函数$f(x) = \hat{\sigma} * x + \hat{\mu}$ ,理解为对 $\bar{x}$ 做了上述操作得到最终的 $x$ , 故可以如上式所言进行变换。

不难发现，我们的回归模型的学习是完全依赖于流模型变换的结果的，而在模型训练的初期，我们学到的变换函数是非常不准的，如果让回归模型去拟合基于不正确的变换得到的目标分布，对于模型性能是有害的。但端到端训练的原则让我们不想将训练变成二阶段的（即先训练流模型，训好以后再训练回归模型），因此本文又引入了一个残差对数似然估计（Residual Log-likelihood Estimation, RLE）的概念。

### Residual with Reparameterization

![image-20230911151306043](https://s2.loli.net/2023/09/11/eYUmFNxh4TJ9LBl.png)
