from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from source.tree_export import ExportModel

"""加载数据集
"""
# 加载iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
feature_names = iris.feature_names
target_names = iris.target_names

"""训练
"""
dt = DecisionTreeClassifier(criterion="gini")   # cart分类树
model = dt.fit(X_train, y_train)
print("model=", model)
y_pred = dt.predict(X_test)
print("y_real=", y_test)
print("y_pred=", y_pred)
cnt = np.sum([1 for i in range(len(y_test)) if y_test[i]==y_pred[i]])
print("right={0}, all={1}".format(cnt, len(y_test)))
print("accuracy={}%".format(100.0 * cnt / len(y_test)))
ExportModel().export_desiontree_to_file(dt, ".\\result_pic\DecisionTreeClassifier", "png", feature_names, target_names, max_depth=None)

"""剪枝
"""
# 1.计算CCP剪枝的α值
model_prune = dt.cost_complexity_pruning_path(X_train, y_train)
cpp_alphas = model_prune['ccp_alphas']
print("model_prune['ccp_alphas']=", cpp_alphas)

# 2.计算不同α值对应的训练集预测准确率和测试集预测准确率
accuracy_train,accuracy_test = [],[]
for v in cpp_alphas:
    tree = DecisionTreeClassifier(criterion="gini", ccp_alpha=v)

    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))

# 3.绘图分析α值与训练集预测准确率和测试集预测准确率的关系
plt.figure(1)
plt.xlabel('cpp_alpha')
plt.ylabel('accuracy')
plt.plot(cpp_alphas, accuracy_train, label='accuracy_train')   # 设置曲线的类型
plt.plot(cpp_alphas, accuracy_test, color='red', linewidth=1.0, linestyle='--', label='accuracy_test')
plt.legend(loc='upper right')  # 绘制图例
plt.show()

# 4.选取最优的α值训练模型并且可视化
dt_prune = DecisionTreeClassifier(criterion="gini", ccp_alpha=0.01428571)
dt_prune.fit(X_train, y_train)
y_pred = dt_prune.predict(X_test)
print("y_real=", y_test)
print("y_pred=", y_pred)
cnt = np.sum([1 for i in range(len(y_test)) if y_test[i]==y_pred[i]])
print("right={0},all={1}".format(cnt, len(y_test)))
print("accury={}%".format(100.0*cnt/len(y_test)))
ExportModel().export_desiontree_to_file(dt_prune, ".\\result_pic\DecisionTreeClassifierPrune", "png",
                                        feature_names, target_names, max_depth=None)

"""结束
"""
print("Finished.")
