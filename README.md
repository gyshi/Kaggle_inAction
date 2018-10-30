<<<<<<< HEAD
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
 
 实验结果：
 
| 算法| 训练集精度|k折交叉验证|
|---|---|---|
| GaussianNB| 0.8601|0.86|
| SVM| 0.9754 |0.96 | 
| KNN| 0.9808|0.97| 
| DecisionTree| 1.0|0.81| 
| Random Forest|0.9991|0.88| 

 
