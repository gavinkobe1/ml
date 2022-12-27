# 引入相关的包
import pandas as pd  # 表格和数据操作
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# -------------------------------------------------------------------------------------
# 读入数据
birth = pd.read_csv(r'./test/daily-total-female-births.csv', index_col=['Date'], parse_dates=['Date'])
birth.info()
birth.head()
plt.figure(figsize=(15, 7))
plt.plot(birth)

# -------------------------------------------------------------------------------------
window = 3
# trail-rolling average transform
rolling = birth.rolling(window=window)
rolling_mean = rolling.mean()
plt.figure(figsize=(15, 7))
plt.plot(birth)
plt.plot(rolling_mean, 'r')

rolling_mean.head()

# -------------------------------------------------------------------------------------
lag1 = birth.shift(1)
lag3 = birth.shift(3)
lag3_mean = lag3.rolling(window=window).mean()
df = pd.concat([lag3_mean, lag1, lag3], axis=1)
df.columns = ['lag3_mean', 't-1', 't-3']

df.head()

# -------------------------------------------------------------------------------------
# prepare situation
X = birth.values
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = []

# walk forward over time steps in test
for t in range(len(test)):
    length = len(history)
    yhat = np.mean([history[i] for i in range(length - window, length)])
    predictions.append(yhat)
    history.append(test[t])
    print('predicted=%f, excepted=%f' % (yhat, test[t]))

error = mean_squared_error(test, predictions)
print('Test MSE: %3f' % error)
plt.figure(figsize=(15, 7))
plt.plot(test)
plt.plot(predictions, 'r')
