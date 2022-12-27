
from source.C45Classifier import C45Classifier
from source.tree_plotter import tree_plot
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

"""加载数据集
"""
# 加载iris数据集
# iris数据集为一个用于识别鸢尾花的机器学习数据集
# 通过四种特征(花瓣长度,花瓣宽度,花萼长度,花萼宽度)来实现三种鸢尾花的类别划分
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
feature_names = np.array(iris.feature_names)

"""创建决策树对象
"""
dt = C45Classifier(use_gpu=True)

"""训练
"""
model = dt.train(X_train, y_train, feature_names)
print("model=", model)
y_pred = dt.predict(X_test)
print("y_real=", y_test)
print("y_pred=", y_pred)
cnt = np.sum([1 for i in range(len(y_test)) if y_test[i]==y_pred[i]])
print("right={0},all={1}".format(cnt, len(y_test)))
print("accury={}%".format(100.0*cnt/len(y_test)))

"""绘制
"""
tree_plot(model)

"""剪枝
"""
model_prune = dt.pruning_rep(X_train, y_train, X_test, y_test)
print("model_prune=", model_prune)
y_pred = dt.predict(X_test)
print("y_real=", y_test)
print("y_pred=", y_pred)
cnt = np.sum([1 for i in range(len(y_test)) if y_test[i]==y_pred[i]])
print("right={0},all={1}".format(cnt, len(y_test)))
print("accury={}%".format(100.0*cnt/len(y_test)))
tree_plot(model_prune)

"""结束
"""
print("Finished.")