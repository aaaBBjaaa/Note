# Pose Estimation Evaluation Methods



### PCK

##### Percentage of Correct Keypoints

计算检测的关键点与其对应的groundtruth间的归一化距离小于设定阈值的比例。FLIC 中是以躯干直径(torso size) 作为归一化参考。MPII 中是以头部长度(head length) 作为归一化参考，即PCKh。
$$
\begin{aligned}PCK_i^k&=\frac{\sum_p\delta(\frac{d_{pi}}{d_p^{def}}\leq T_k)}{\sum_p1}\\\\\\PCK_{mean}^k&=\frac{\sum_p\sum_i\delta(\frac{d_{pi}}{d_p^{def}}\leq T_k)}{\sum_p\sum_i1}\end{aligned}
$$
其中 $i$ 表示id为 $i$ 的关键点

$k$ 表示第k个阈值 $T_{k}$

$d_{pi}$表示第p个行人

$d_{p}^{i}$ 表示第 $p$个人中id为 $i$ 的关键点的预测值和groundtruth的欧式距离

$d_{p}^{def}$ 表示第 $p$ 个人的尺度因子，这个因子不同公开数据集使用的计算方法不同，就是上文提到的FLIC和MPII数据集的差别。

 $T_{k}$ 表示人工设定的阈值，  $T_{k}$∈[0:0.01:0.1]

$PCK_{i}^{k}$ 表示  $T_{k}$ 阈值下id为i的关键点的PCK指标

$PCK_{mean}^{k}$ 表示 $T_{k}$阈值下的算法PCK指标



### OCK

##### object keypoints similarity

关键点相似度，在人体关键点评价任务中,对于网络得到的关键点好坏,并不是仅仅通过简单的欧氏距离来计算的,而是有一定的尺度加入,来计算两点之间的相似度。
$$
OKS_p=\frac{\sum_iexp^{-d_{p^{i}}^2/2S_p^2\sigma_i^2}\delta(v_{p^{i}}>0)}{\sum_i\delta(v_{p^{i}}>0)}
$$
其中：

$p$ 表示在groundtruth中某个人

$p_{i}$ 表示表示某个人的关键点

$d_{p^{i}}$ 表示当前检测的一组关键点中id为$i$的关键点与groundtruth里行人p的关键点中id为 $i$ 的关键点的欧式距离 $d_{p}^{i}$=$\sqrt{(x_{i}'−x_{p^{i}})(y_{i}'−y_{p^{i}})}$ ， 

($x_{i}′$,$y_{i}′$) 为当前的关键点检测结果， ($x_{p^{i}}$,$y_{p^{i}}$) 是groundtruth里行人p的关键点id为i的关键点。

$v_{p^{i}}$=1 表示这个关键点的可见性为1，即关键点无遮挡并且已标注。 $v_{p^{i}}$=2 表示关键点有遮挡但已标注。

$S_{p}$ 表示groundtruth行人中id为p的人的尺度因子，其值为行人检测框面积的平方根： $S_{p}$=$\sqrt{wh}$ ， $w$,$h$ 为检测框的宽和高，这里的检测框就是bounding box。

$\sigma_{i}$ 表示id为i的关键点归一化因子，这个因子是通过对所有的样本集中的groundtruth关键点由人工标注与真实值存在的标准差， $\sigma$ 越大表示此类型的关键点越难标注。对**coco**数据集中的**5000**个样本统计出**17**类关键点的归一化因子， $\sigma$ 的取值可以为：**{鼻子：0.026，眼睛：0.025，耳朵：0.035，肩膀：0.079，手肘：0.072，手腕：0.062，臀部：0.107，膝盖：0.087，脚踝：0.089}**，因此可以当作常数看待，但是使用的类型仅限这个里面。如果使用的关键点类型不在此当中，可以使用另外一种统计方法计算此值，详细见下文

$\delta(*)$表示如果条件成立为1，否则为0 ，在此处的含义是：仅计算groundtruth中已标注的关键点。

### ACC

##### Accuracy

**单人姿态估计ACC：**

计算出groundtruth与检测得到的关键点的相似度**OKS or PCK**为一个标量，然后人为的给定一个阈值**T**，然后可以通过所有图片的**OKS or PCK**计算**AP**:
$$
AP=\frac{\sum_p\delta(oks / pck_p>T)}{\sum_p1}
$$


**多人姿态估计ACC：**

- 如果采用的检测方法是自顶向下，先把所有的人找出来再检测关键点，那么其**AP**计算方法如同**单人姿态估计AP**

- 如果采用的检测方法是自底向上，先把所有的关键点找出来然后再组成人。假设一张图片中有M个人，预测出N个人，由于不知道预测出的**N**个人与groundtruth中的**M**个人的一一对应关系，因此需要计算groundtruth中每一个人与预测的**N**个人的**oks**，那么可以获得一个大小为*M*×*N*的矩阵，矩阵的每一行为groundtruth中的一个人与预测结果的**N**个人的**oks**，然后找出每一行中**oks**最大的值作为当前**GT**的**oks**。最后每一个**GT**行人都有一个标量**oks**，然后人为的给定一个阈值**T**，然后可以通过所有图片中的所有行人计算**AP**:
  $$
  AP=\frac{\sum_{m}\sum_{p}\delta(oks/pck_p>T)}{\sum_{m}\sum_{p}1}
  $$

**MAP** :

mAP是常用检测指标，具体就是给**AP**指标中的人工阈值**T**设定不同的值，然后会获得多个**AP**指标，最后再对多个**AP**指标求平均，最终获得mAP。

### NME

##### Normalized Mean Error


$$
NME\big(P,\widehat{P}\big)=\frac{1}{M}\sum_{i=1}^{M}\frac{|\big|P_{i}-\widehat{P}_{i}\big||_{2}}{d}
$$
其中：

$\widehat{P}_i$表示该第$i$个关键点的预测坐标，$P_{i}$表示真实坐标

$M$表示的是关键点的总个数 

$d$是人为选定的距离，可以是眼间距，瞳间距，也可以是人体\人脸\手部的外接矩形的斜距



### 混淆矩阵

通常用在MAP中，在关键点检测中可以使用，但不常见。关键点检测必须有confidence这一预测值才可能回出现混淆矩阵。confidence的人为设定阈值来筛选结果就等同于OD中的NMS。其置信度大于设定阈值的预测值代表predict positive。而进行PCK计算大于某一个设定阈值才能证明是GT positive。

![image-20230917151921362](https://s2.loli.net/2023/09/17/E3LID8ZByPGeXu1.png)



通俗解释：从Object Detection出发

**TP** (True Positive)：一个正确的检测，检测的IOU ≥ *threshold*。即预测的边界框(bounding box)中分类正确且边界框坐标正确的数量。

**FP** (False Positive)：一个错误的检测，检测的IOU < *threshold*。即预测的边界框中分类错误或者边界框坐标不达标的数量，即预测出的所有边界框中除去预测正确的边界框，剩下的边界框的数量。

**FN** (False Negative)：一个没有被检测出来的ground truth。所有没有预测到的边界框的数量，即正确的边界框(ground truth)中除去被预测正确的边界框，剩下的边界框的数量。

**Precision** (准确率 / 精确率)：「Precision is the ability of a model to identify **only** the relevant objects」，准确率是模型**只找到**相关目标的能力，等于TP/(TP+FP)。即模型给出的所有预测结果中命中真实目标的比例。

**Recall** (召回率)：「Recall is the ability of a model to find all the relevant cases (all ground truth bounding boxes)」，召回率是模型**找到所有**相关目标的能力，等于TP/(TP+FN)。即模型给出的预测结果最多能覆盖多少真实目标。

### AUC

##### Area under curve

值得一提此处的AUC的计算完全依赖于纵坐标，这个纵坐标在关键点检测中可以是ACC。而不一定是ROC曲线面积。