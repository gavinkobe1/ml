from source.CartClassifier import CartClassifier
from source.tree_plotter import tree_plot
import numpy as np
import csv

""" 加载数据集
"""
# 加载 play_tennis 数据集
with open("data/play_tennis.csv", "r", encoding="gbk") as f:
    text = list(csv.reader(f))
    x_names = np.array(text[0][:-1])
    y_names = text[0][-1]
    x = np.array([v[:-1] for v in text[1:]])
    y = np.array([v[-1] for v in text[1:]])
    x_train, x_test, y_train, y_test = x, x, y, y

""" 创建决策树对象
"""
dt = CartClassifier(use_gpu=True)

""" 训练模型
"""
model = dt.train(x_train, y_train, x_names)
print("model =", model)

""" 预测
"""
y_pred = dt.predict(x_test)
print("y_real", y_test)
print("y_pred", y_pred)
cnt = np.sum([1 for i in range(len(y_test)) if y_test[i]==y_pred[i]])
print("right={0}, all={1}".format(cnt, len(y_test)))
print("accury={}%".format(100.0 * cnt / len(y_test)))

"""绘制
"""
tree_plot(model)

"""剪枝
"""
model_prune = dt.pruning(x_train, y_train)
print("model_prune=", model_prune)
y_pred = dt.predict(x_test)
print("y_real=", y_test)
print("y_pred=", y_pred)
cnt = np.sum([1 for i in range(len(y_test)) if y_test[i]==y_pred[i]])
print("right={0}, all={1}".format(cnt, len(y_test)))
print("accury={}%".format(100.0 * cnt / len(y_test)))

"""绘制
"""
tree_plot(model_prune)

"""结束
"""
print("Finished")
