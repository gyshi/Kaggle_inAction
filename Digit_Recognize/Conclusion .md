
# digit recognizer

我所采用的数据集为kaggle的竞赛Digit Recognizer的数据集，
我分别采用了朴素贝叶斯、KNN、k-means、SVM、Decision tree
、Random Forest
对比了和使用了PCA降维之后的准确度的比较。
实验步骤如下：

 - 读取数据
 - 数据预处理
 - 模型选择
 - 模型优化
 

# 未使用PCA

下面为不使用PCA的结果（只进行了部分的测试集验证）：
[代码](https://github.com/1360024032/kaggle_inaction/blob/master/Digit_Recognize/Method_1.py)

 - Naive Bayes:
  I use **GaussianNB**,on the train set, the accuracy is **0.55719047619**, it takes 17.858203172683716 seconds
  the accuracy is **0.56(+- 0.01)** 
  I use **MultinomialNB**,on the train set, the accuracy is **0.82480952381**, it takes 2.062295913696289 seconds
  with the k-fold ,the accuracy is **0.82(+- 0.01)**
  I use **BernoulliNB**,on the train set, the accuracy is **0.834785714286**, it takes 1.9217422008514404 seconds
  with the k-fold ,the accuracy is **0.83(+- 0.01**) the test set is **0.83242**
-  SVM
  I use **SVM**,on the train set, the accuracy is **0.9405**, it takes 15.45 minutes,with the k-fold ,the accuracy is **0.93(+- 0.01)** the test set is **0.93600**
  - KNN
  I use KNN,on the train set, the accuracy is **0.97142857143**, it takes 38.07 minutes， with the k-fold ,the accuracy is **0.97(+- 0.00)** the test set is **0.97000**
  - Kmeans
I use **Kmeans**,on the train set, the accuracy is **0.14830952381**, it takes 1.877 minutes
  with the k-fold ,the accuracy is -340919.55(+- 1785.45)，the test set is **0.15028**
  -  The DecisionTree
I use **DecisionTree**,on the train set, the accuracy is **1.0,** it takes 12.842848539352417 seconds,  with the k-fold ,the accuracy is **0.85(+- 0.00)**
  - Random Forest
I use **Random Forest** ,on the train set, the accuracy is **0.999**,with the k-fold ,the accuracy is **0.94(+- 0.00)**

# 使用PCA

 - Naive Bayes:
 I use **GaussianNB**,on the train set, the accuracy is **0.860119047619**,   the accuracy is **0.86(+- 0.01)** 

 - SVM
 I use **SVM**,on the train set, the accuracy is ** 0.975452380952**,   the accuracy is **0.96(+- 0.00)** 
 -   KNN
I use **KNN**,on the train set, the accuracy is **0.980761904762**。with the k-fold ,the accuracy is **0.97(+- 0.00)**  the test set is **0.97000**
 -   Kmeans
I use **Kmeans**,on the train set, the accuracy is **0.0858095238095,** with the k-fold ,the accuracy is -318020.78(+- 1645.13)
 -   The DecisionTree
I use **DecisionTree**,on the train set, the accuracy is **1.0,** 
  with the k-fold ,the accuracy is **0.81(+- 0.01)**
 - Random Forest
I use Random Forest ,on the train set, the accuracy is **0.999119047619**, with the k-fold ,the accuracy is 0.88(+- 0.01) the test set is **0.8800**


可以观察 通过k折交叉验证，其精度和在测试集上的精度基本一致。

# 算法分析：
  数字识别数据是离散的，是分类问题，不是连续的，所以不要用回归方法。下面主要介绍几种算法的特点：
-  Naive Bayes
  朴素贝叶斯属于监督学习的生成模型，能处理生成模型，可以处理多类别问题，计算量小，采用了属性条件独立性假设。对连续的属性采用了概率密度函数，采用了高斯分布。三种贝叶斯方法：
 其中GaussianNB就是先验为高斯分布的朴素贝叶，MultinomialNB就是先验为多项式分布的朴素贝叶斯，而BernoulliNB就是先验为伯努利分布的朴素贝叶斯。
一般来说，如果样本特征的分布大部分是连续值，使用GaussianNB会比较好。如果如果样本特征的分大部分是多元离散值，使用MultinomialNB比较合适。而如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用BernoulliNB。

-  SVM
SVM在中小量样本规模的时候容易得到数据和特征之间的非线性关系，可以避免使用神经网络结构选择和局部极小值问题，可解释性强，可以解决高维问题。 
SVM对缺失数据敏感，对非线性问题没有通用的解决方案，核函数的正确选择不容易，计算复杂度高，主流的算法可以达到O(n2)O(n2)的复杂度，这对大规模的数据是计算量很大。

-   KNN
监督学习方法，对于给定样本，根据距离度量寻找训练集中与其最靠进的k个训练样本。精度高，对异常值不敏感，但是计算复杂度高。

- Kmeans
无监督学习方法，主要用于连续属性的计算，容易实现，但是k值的选择很重要。

- The Decision Tree
 一种分类算法， 计算复杂度不高，可以处理不相关的特征数据，泛化能力强，适用于离散和连续的属性值，但是容易过拟合。

- Random Forest
集成学习方法，它以决策树为基学习器构建的，在决策树的训练过程引入了随机属性的选择。容易实现，计算开销小。

# 结果分析
没有用PCA降维的算法

| 算法| 训练集精度|k折交叉验证| 时间|
|---|---|---|---|
| GaussianNB| 0.5572|0.56| 17.86s|
| MultinomialNB| 0.8248|0.82|  2.06s|
| BernoulliNB| 0.8348|0.83| 1.92s|
| SVM| 0.9405 |0.93 | 15.45 min|
| KNN| 0.9714|0.97| 38.07 min|
| Kmeans| 0.1483|-34091| 1.82min|
| DecisionTree| 1.0| 0.85| 12.84s|
| Random Forest| 0.999|0.94| 4.08s|

从结果可以看出kmeans和GaussianNB结果非常差劲，表现不好，因为这两个算法适用于连续性的属性计算。在k折交叉验证中，knn表现得结果最好，但是花费的时间也是最久的，因为每一次都要计算与k个样本的距离，需要大量的算力，消耗计算资源。
我们发现决策树在训练集中的精度和k折交叉验证中的精度相差很大，原因是决策时是根据最优的属性进行分类，导致了过拟合。
表现较好的是随机森林、svm、knn。
下面我们进行了PCA聚类，除去冗余的特征，只考虑准确率。

| 算法| 训练集精度|k折交叉验证|
|---|---|---|
| GaussianNB| 0.8601|0.86|
| SVM| 0.9754 |0.96 | 
| KNN| 0.9808|0.97| 
| DecisionTree| 1.0|0.81| 
| Random Forest|0.9991|0.88| 

进行pca降维后，特征值达到了154，原始特征为784。对比没有进行pca降维的结果，我们可以发现GaussianNB、SVM均有精度的提高，knn基本不变。因为PCA除去了冗余的特征，保留了主要的特征。但是随机森林的精度下降了，因为这里和决策树一样，去除了许多属性，因为其基学习器决策树本来就是依据属性来进行决策的，现在属性减少，精度也会下降。因为随机森林就是从当前节点随机选包含k个属性的子集，在从子集中选择一个最优属性划分，现在属性减少，随机性也会减小。

GaussianNB的增加因为进行PCA将维后，使得样本投影的方差放缩到单位方差，对于贝叶斯高斯分布来说，使得数据点更比原来更“连续”了。
现在为了达到更好的结果，我对SVM做优化，来获得最优的准确性。通过PCA的维数设置，使得特征下降到了46个，得到了0.98242的精度。

[PCA+SVM代码](https://github.com/1360024032/kaggle_inaction/blob/master/Digit_Recognize/PCA_SVM.py)






  
  
  
