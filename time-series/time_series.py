# 引入相关的统计包
import warnings  # 忽略警告

warnings.filterwarnings('ignore')

import numpy as np  # 矢量和矩阵
import pandas as pd  # 表格和数据操作
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta  # 有风格地处理日期
from scipy.optimize import minimize  # 函数优化
import statsmodels.formula.api as smf  # 统计与经济计量
import statsmodels.tsa.api as smt
import scipy.stats as scs
from itertools import product
from tqdm import tqdm_notebook
import statsmodels.api as sm

#=====================================================================================

# 1 如真实的手机游戏数据，将调查每小时观看的广告和每天花费的游戏币
ads = pd.read_csv(r'./test/ads.csv', index_col=['Time'], parse_dates=['Time'])
currency = pd.read_csv(r'./test/currency.csv', index_col=['Time'], parse_dates=['Time'])

#=====================================================================================

# 2 绘图，查看数据情况
# 广告
ads.info()  # 查看数据的基本情况：多少条记录、是否有缺失值、列名等
plt.figure(figsize=(15, 7))
plt.plot(ads.Ads)
# plt.xticks(rotation=30, fontsize=10)
plt.title('Ads watched (hourly data)')
plt.grid(True)
plt.show()
#-------------------------------------------------------------------------------------
# 货币
currency.info()
plt.figure(figsize=(15, 7))
plt.plot(currency.GEMS_GEMS_SPENT)
plt.title('In-game currency spent (daily data)')
plt.grid(True)
plt.show()

#=====================================================================================

# 3 预测质量指标（forecast quality metrics),引入相关指标
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#=====================================================================================

# 4 预测方法 ：移动、平滑和估计（move,smooth,evaluate))
# 4.1 移动：简单移动平均法
def moving_average(series, n):
    """移动平均计算最近n个观察值的均值"""
    return np.average(series[-n:])

#-------------------------------------------------------------------------------------
moving_average(ads, 24)

#-------------------------------------------------------------------------------------
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
    series: dataframe with timeseries
    window:rolling window size
    plot_intervals:show confidence interval
    plot_anomalies:show anomalies
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title('Moving average\n window size={}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')

    # plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, 'r--', label='Upper Bond / Lower Bond')
        plt.plot(lower_bond, 'r--')

    # Having the intervals, find abnormal values
    if plot_anomalies:
        anomalies = pd.DataFrame(index=series.index, columns=series.columns)
        anomalies[series < lower_bond] = series[series < lower_bond]
        anomalies[series > upper_bond] = series[series > upper_bond]
        plt.plot(anomalies, 'ro', markersize=10)

    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='upper left')
    plt.grid(True)

#-------------------------------------------------------------------------------------
# 绘制不同window下的移动平均值和真实值，会发现随着window的增加，移动平均曲线更加平滑
plotMovingAverage(ads, 4) # smooth by the previous 4 hours
plotMovingAverage(ads, 12) # smooth by the previous 12 hours
plotMovingAverage(ads, 24) # smooth by the previous 24 hours, get daily trend
plotMovingAverage(ads, 4, plot_intervals=True)  # 绘制置信区间，查看是否有异常值

#-------------------------------------------------------------------------------------
# 曲线基本正常，故意创造一个含异常值的序列 ads_anomaly
ads_anomaly = ads.copy()
ads_anomaly.iloc[-20] = ads_anomaly.iloc[-20] * 0.2
plotMovingAverage(ads_anomaly, 4, plot_intervals=True, plot_anomalies=True)

#-------------------------------------------------------------------------------------
# 第二个序列
plotMovingAverage(currency, 7, plot_intervals=True, plot_anomalies=True)

#=====================================================================================

# 4.1 移动：加权平均，移动平均未捕捉到数据中的月度季节性，
# 并将几乎所有30天的峰值标记为异常。如果想避免误报，最好考虑更复杂的模型。
def weighted_average(series, weights):
    """计算序列的加权平均"""
    result = 0.0
    for n in range(len(weights)):
        result += series.iloc[-n - 1] * weights[n]
    return float(result)

#-------------------------------------------------------------------------------------
weighted_average(ads, [0.6, 0.3, 0.1])

#=====================================================================================

# 4.2 平滑：一次指数平滑法
def exponential_smoothing(series, alpha):
    """
    series:dataset with timestamps
    alpha:float [0.0,1.0], smoothing parameter
    """
    result = [series[0]]  # first values is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result


def plotExponentialSmoothing(series, alphas):
    """
    Plot exponential smoothing with different alphas
    series:dataset with timestamps
    alpha:list of floats, smoothing parameters
    """
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(15, 7))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label='Alpha {}'.format(alpha))
        plt.plot(series.values, 'c', label='Actual')
        plt.legend(loc='best')
        plt.axis('tight')
        plt.title('Exponential Smoothing')
        plt.grid(True)

#-------------------------------------------------------------------------------------
plotExponentialSmoothing(ads.Ads, [0.3, 0.05])
plotExponentialSmoothing(currency.GEMS_GEMS_SPENT, [0.3, 0.05])

#=====================================================================================

# 4.2 平滑：双指数平滑法
def double_exponential_smoothing(series, alpha, beta):
    """
    series:dataset with timeseries
    alpha:float[0.0,1.0], smoothing parameter for level
    beta:float[0.0,1.0], smoothing parameter for trend
    """
    #  first value is same as series
    result = [series[0]]
    for n in range(1, len(series) + 1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def plotDoubleExponentialSmoothing(series, alphas, betas):
    """
    Plot double exponential smoothing with different alphas and betas
    series:dataset with timestamps
    alphas:list of floats, smoothing parameters for level
    betas:list of floats, smoothing parameters for trend
    """
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(15, 7))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta),
                         label='Alpha {}, Beta {}'.format(alpha, beta))
        plt.plot(series.values, label='Actual')
        plt.legend(loc='best')
        plt.axis('tight')
        plt.title('Double Exponential Smoothing')
        plt.grid(True)

#-------------------------------------------------------------------------------------
plotDoubleExponentialSmoothing(ads.Ads, alphas=[0.9, 0.02], betas=[0.9, 0.02])
plotDoubleExponentialSmoothing(currency.GEMS_GEMS_SPENT, alphas=[0.9, 0.02], betas=[0.9, 0.02])

#=====================================================================================

# 4.2 平滑：三次指数平滑法
class HoltWinters:
    """
    Holt-Winter model with the anomalies detection using Brutlag method
    series:initial time series
    slen:length of a season
    alpha, beta, gamma:Holt-Winter model coefficients
    n_preds:predictions horizon
    scaling_factor:sets the width of the confidence interval by Brutlag (usually take values from 2 to 3)
    """

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor

    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # 计算每个季节平均值
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))
        # 开始计算初始值
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:  # 成分初始化
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(self.result[0] + self.scaling_factor * self.PredictedDeviation[0])
                self.LowerBond.append(self.result[0] - self.scaling_factor * self.PredictedDeviation[0])

                continue

            if i >= len(self.series):  # 预测
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])

                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha * (val - seasonals[i % self.slen]) \
                                      + (1 - self.alpha) * (smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Deviation is calculated according to Brutlag algorithm
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i])
                                               + (1 - self.gamma) * self.PredictedDeviation[-1])

            self.UpperBond.append(self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1])
            self.LowerBond.append(self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])

#-------------------------------------------------------------------------------------
def plotHoltWinters(series, plot_intervals=False, plot_anomalies=False):
    """
    series:dataset with timeseries
    plot_interval:show confidence intervals
    plot_anomalies:show anomalies
    """

    plt.figure(figsize=(20, 10))
    plt.plot(model.result, label='Model')
    plt.plot(series.values, label='Actual')
    error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
    plt.title('Mean Absolute Percentage Error: {0:.2f}%'.format(error))

    if plot_anomalies:
        anomalies = np.array([np.NaN] * len(series))
        anomalies[series.values < model.LowerBond[:len(series)]] = \
            series.values[series.values < model.LowerBond[:len(series)]]
        anomalies[series.values > model.UpperBond[:len(series)]] = \
            series.values[series.values > model.UpperBond[:len(series)]]
        plt.plot(anomalies, 'o', markersize=10, label='Anomalies')

    if plot_intervals:
        plt.plot(model.UpperBond, 'r--', alpha=0.5, label='Up/Low confidence')
        plt.plot(model.LowerBond, 'r--', alpha=0.5)
        plt.fill_between(x=range(0, len(model.result)), y1=model.UpperBond,
                         y2=model.LowerBond, alpha=0.2, color='lightgrey')
    plt.vlines(len(series), ymin=min(model.result), ymax=max(model.UpperBond), linestyles='dashed')
    plt.axvspan(len(series) - 20, len(model.result), alpha=0.3, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='best', fontsize=13)

#-------------------------------------------------------------------------------------
# 时序交叉验证
from sklearn.model_selection import TimeSeriesSplit


def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=24):
    """
    Return error on CV
    params:vector of parameters for optimization
    series:dataset with timeseries
    slen:season length for Holt-Winters model
    """
    errors = []
    values = series.values
    alpha, beta, gamma = params

    tscv = TimeSeriesSplit(n_splits=3)

    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):
        model = HoltWinters(series=values[train], slen=slen, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()
        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
    return np.mean(np.array(errors))

#-------------------------------------------------------------------------------------
%%time
data = ads.Ads[:-20]  # leave some data for test

##初始化模型参数 alpha beta gamma
x = [0, 0, 0]

# Minimizing the loss function， 优化方法 Newton Conjugate-Gradient
opt = minimize(timeseriesCVscore, x0=x, args=(data, mean_squared_log_error), method='TNC', bounds=(
    (0, 1), (0, 1), (0, 1)))

# take the optimal values
alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)

# train the model with them, forecasting for the next 50 hours, 24-hour seasonality
model = HoltWinters(data, slen=24, alpha=alpha_final, beta=beta_final, gamma=gamma_final, n_preds=50, scaling_factor=3)
model.triple_exponential_smoothing()

#-------------------------------------------------------------------------------------
plotHoltWinters(ads.Ads)
plotHoltWinters(ads.Ads, plot_intervals=True, plot_anomalies=True)

#-------------------------------------------------------------------------------------
plt.figure(figsize=(25,5))
plt.plot(model.PredictedDeviation)
plt.grid(True)
plt.axis('tight')
plt.title("Brutlag's predicted deviation" )

#-------------------------------------------------------------------------------------
##应用到第二个序列，30天的季节性
%%time
data= currency.GEMS_GEMS_SPENT[:-50]
slen=30 # 30-day seasonality
x=[0,0,0]
opt=minimize(timeseriesCVscore, x0=x,args=(data,mean_absolute_percentage_error, slen),
             method='TNC',bounds=((0,1),(0,1),(0,1)))
alpha_final,beta_final,gamma_final= opt.x
print(alpha_final,beta_final,gamma_final)

model=HoltWinters(data,slen=slen,alpha=alpha_final,beta=beta_final,gamma=gamma_final,n_preds=100,scaling_factor=3)
model.triple_exponential_smoothing()

#-------------------------------------------------------------------------------------
plotHoltWinters(currency.GEMS_GEMS_SPENT)
plotHoltWinters(currency.GEMS_GEMS_SPENT, plot_intervals=True,plot_anomalies=True)

#-------------------------------------------------------------------------------------
plt.figure(figsize=(20,5))
plt.plot(model.PredictedDeviation)
plt.grid(True)
plt.axis('tight')
plt.title("Brutlag's predicted deviation")

#=====================================================================================

# 5经济计量方法（Econometric approach)
# ARIMA 属于经济计量方法
# 创建平稳序列
white_noise = np.random.normal(size=1000)
with plt.style.context('bmh'):
    plt.figure(figsize=(15,5))
    plt.plot(white_noise)

#-------------------------------------------------------------------------------------
def plotProcess(n_samples=1000,rho=0):
    x=w=np.random.normal(size=n_samples)
    for t in range(n_samples):
        x[t] = rho*x[t-1]+w[t]

    with plt.style.context('bmh'):
        plt.figure(figsize=(10,5))
        plt.plot(x)
        plt.title('Rho {}\n Dickey-Fuller p-value: {}'.format(rho,round(sm.tsa.stattools.adfuller(x)[1],3)))

#-------------------------------------------------------------------------------------
for rho in [0,0.6,0.9,1]:
    plotProcess(rho=rho)

#=====================================================================================

# 摆脱非平稳性，建立SARIMA（Getting rid of non-stationarity and building SARIMA）

def tsplot(y,lags=None,figsize=(12,7),style='bmh'):
    """
    Plot time series, its ACF and PACF, calculate Dickey-Fuller test
    y:timeseries
    lags:how many lags to include in ACF,PACF calculation
    """
     if not isinstance(y, pd.Series):
         y = pd.Series(y)

     with plt.style.context(style):
         fig = plt.figure(figsize=figsize)
         layout=(2,2)
         ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
         acf_ax = plt.subplot2grid(layout, (1,0))
         pacf_ax = plt.subplot2grid(layout, (1,1))

         y.plot(ax=ts_ax)
         p_value = sm.tsa.stattools.adfuller(y)[1]
         ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
         smt.graphics.plot_acf(y,lags=lags, ax=acf_ax)
         smt.graphics.plot_pacf(y,lags=lags, ax=pacf_ax)
         plt.tight_layout()

#-------------------------------------------------------------------------------------
tsplot(ads.Ads, lags=60)

#-------------------------------------------------------------------------------------
ads_diff = ads.Ads-ads.Ads.shift(24) # 去除季节性
tsplot(ads_diff[24:], lags=60)

ads_diff = ads_diff - ads_diff.shift(1) # 最终图
tsplot(ads_diff[24+1:], lags=60)

#=====================================================================================

# 建模 SARIMA
# setting initial values and some bounds for them
ps = range(2,5)
d=1
qs=range(2,5)
Ps=range(0,2)
D=1
Qs=range(0,2)
s=24 #season length

# creating list with all the possible combinations of parameters
parameters=product(ps,qs,Ps,Qs)
parameters_list = list(parameters)
print(parameters)
print(parameters_list)
print(len(parameters_list))

#-------------------------------------------------------------------------------------
def optimizeSARIMA(parameters_list, d,D,s):
    """
    Return dataframe with parameters and corresponding AIC
    parameters_list:list with (p,q,P,Q) tuples
    d:integration order in ARIMA model
    D:seasonal integration order
    s:length of season
    """
    results = []
    best_aic = float('inf')

    for param in tqdm_notebook(parameters_list):
        # we need try-exccept because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(ads.Ads, order=(param[0], d,param[1]),
                                              seasonal_order=(param[2], D,param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic<best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic',ascending=True).reset_index(drop=True)

    return result_table

#------------------------------------------------------------------------------------
%%time
result_table = optimizeSARIMA(parameters_list, d,D,s)

#-------------------------------------------------------------------------------------
result_table.head()

# set the parameters that give the lowerst AIC
p,q,P,Q = result_table.parameters[0]
best_model = sm.tsa.statespace.SARIMAX(ads.Ads, order=(p,d,q),seasonal_order=(P,D,Q,s)).fit(disp=-1)
print(best_model.summary())

# inspect the residuals of the model
tsplot(best_model.resid[24+1:], lags=60)

#-------------------------------------------------------------------------------------
def plotSARIMA(series, model, n_steps):
    """
    plot model vs predicted values
    series:dataset with timeseries
    model:fitted SARIMA model
    n_steps:number of steps to predict in the future
    """
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model']=model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model due the differentiating
    data['arima_model'][:s+d]=np.nan

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0],end=data.shape[0]+n_steps)
    forecast = data.arima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])

    plt.figure(figsize=(15,7))
    plt.title('Mean Absolute Percentage Error: {0:.2f}%'.format(error))
    plt.plot(forecast,color='r',label='model')
    plt.axvspan(data.index[-1],forecast.index[-1], alpha=0.5,color='lightgrey')
    plt.plot(data.actual,label='actual')
    plt.legend()
    plt.grid(True)

#-------------------------------------------------------------------------------------
plotSARIMA(ads, best_model, 50)

#=====================================================================================

# 其他模型：线性...
# Creating a copy of the initial dataframe to make various transformations
data = pd.DataFrame(ads.Ads.copy())
data.columns=['y']

# Adding the lag of the target variable from 6 setps back up to 24
for i in range(6,25):
    data['lag_{}'.format(i)]=data.y.shift(i)

# take a look at the new dataframe
data.tail(7)

#-------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# for time-series cross-validation set 5 folds
tscv = TimeSeriesSplit(n_splits=5)

def timeseries_train_test_split(X,y,test_size):
    """
    Perform train-test split with respect to time series structure
    """
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    return X_train,X_test,y_train,y_test

#-------------------------------------------------------------------------------------
y = data.dropna().y
X = data.dropna().drop(['y'],axis=1)

# reserve 30% of data for testing
X_train, X_test, y_train, y_test =timeseries_train_test_split(X,y,test_size=0.3)

#-------------------------------------------------------------------------------------
# machine learning in two lines
lr = LinearRegression()
lr.fit(X_train, y_train)

#-------------------------------------------------------------------------------------
def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
    """
    Plot modelled vs fact values, prediction intervals and anomalies
    """
    prediction = model.predict(X_test)

    plt.figure(figsize=(15,7))
    plt.plot(prediction,'g',label='prediction', linewidth=2.0)
    plt.plot(y_test.values, label='actual', linewidth=2.0)

    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
        mae = cv.mean() *(-1)
        deviation = cv.std()

        scale=1.96
        lower = prediction-(mae + scale * deviation)
        upper = prediction + (mae + scale *deviation)

        plt.plot(lower, 'r--', label='upper bond / lower bond', alpha=0.5)
        plt.plot(upper, 'r--', alpha=0.5)

    if plot_anomalies:
        anomalies = np.array([np.nan]*len(y_test))
        anomalies[y_test<lower] = y_test[y_test<lower]
        anomalies[y_test>upper] = y_test[y_test>upper]
        plt.plot(anomalies, 'o', markersize=10, label='Anomalies')

    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title('Mean absolute percentage error {0:.2f}%'.format(error))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.grid(True)


def plotCoefficients(model):
    """
    PLots sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, X_train.columns)
    print()
    coefs.columns = ['coef']
    coefs['abs'] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by='abs', ascending=False).drop(['abs'],axis=1)

    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0,xmin=0, xmax=len(coefs), linestyles='dashed')

#-------------------------------------------------------------------------------------
plotModelResults(lr, plot_intervals=True)
plotCoefficients(lr)

#-------------------------------------------------------------------------------------
# 提取时间特征 hour、day of week、is_weekend
data.index = pd.to_datetime(data.index)
data['hour'] = data.index.hour
data['weekday'] = data.index.weekday
data['is_weekend'] = data.weekday.isin([5,6])*1
data.tail()

#-------------------------------------------------------------------------------------
# 可视化特征
plt.figure(figsize=(16,5))
plt.title('Encoded features')
data.hour.plot()
data.weekday.plot()
data.is_weekend.plot()
plt.grid(True)

#-------------------------------------------------------------------------------------
# 特征的尺度不一样，需要归一化

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)
X_train, X_test, y_train, y_test =timeseries_train_test_split(X,y,test_size=0.3)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

#-------------------------------------------------------------------------------------
plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
plotCoefficients(lr)

#-------------------------------------------------------------------------------------
# 目标编码

def code_mean(data, cat_feature, real_feature):
    """
    Returns a dictionary where keys are unique categories of the cat_feature,
    and values are means over real_feature
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())

average_hour = code_mean(data, 'hour', 'y')
plt.figure(figsize=(7,5))
plt.title('Hour averages')
pd.DataFrame.from_dict(average_hour, orient='index')[0].plot()
plt.grid(True)

#-------------------------------------------------------------------------------------
# 将所有的数据准备结合到一起
def prepareData(series, lag_start, lag_end, test_size, target_encoding=False):
    """
    series: pd.DataFrame or dataframe with timeseries
    lag_start: int, initial step back in time to slice target variable
               example-lag_start=1 means that the model will see yesterday's values to predict today
    lag_end: int, finial step back in time to slice target variable
             example-lag_end=4 means that the model will see up to 4 days back in time to predict today
    test_size:float, size of the test dataset after train/test split as percentage of dataset
    target_encoding:boolean, if True -  add target averages to dataset
    """

    # copy of the initial dataset
    data = pd.DataFrame(series.copy())
    data.columns = ['y']

    # lags of series
    for i in range(lag_start, lag_end):
        data['lag_{}'.format(i)]=data.y.shift(i)

    # datatime features
    data.index = pd.to_datetime(data.index)
    data['hour'] = data.index.hour
    data['weekday'] =data.index.weekday
    data['is_weekend']=data.weekday.isin([5,6])*1

    if target_encoding:
        # calculate averages on train set only
        test_index = int(len(data.dropna())*(1-test_size))
        data['weekday_average'] = list(map(code_mean(data[:test_index], 'weekday', 'y').get, data.weekday))
        data['hour_average'] = list(map(code_mean(data[:test_index], 'hour', 'y').get, data.hour))

        # drop encoded variables
        data.drop(['hour', 'weekday'], axis=1, inplace=True)

    # train-test split
    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

#-------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = prepareData(ads.Ads, lag_start=6, lag_end=25, test_size=0.3, target_encoding=True)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)
plotCoefficients(lr)
# plt.xticks(rotation=45, fontsize=7)

#-------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = prepareData(ads.Ads, lag_start=6, lag_end=25, test_size=0.3, target_encoding=False)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#-------------------------------------------------------------------------------------
# 若过拟合，对特征进行正则化
# 先看一下特征之间的相关性
y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)
X_train, X_test, y_train, y_test =timeseries_train_test_split(X,y,test_size=0.3)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

plt.figure(figsize=(10,8))
sns.heatmap(X_train.corr())

#-------------------------------------------------------------------------------------
# 开始正则化
# 岭回归
from sklearn.linear_model import LassoCV, RidgeCV

ridge = RidgeCV(cv=tscv)
ridge.fit(X_train_scaled,y_train)
plotModelResults(ridge, X_train=X_train_scaled,X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)
plotCoefficients(ridge)

# 套索回归
lasso = LassoCV(cv=tscv)
lasso.fit(X_train_scaled,y_train)
plotModelResults(lasso, X_train=X_train_scaled,X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)
plotCoefficients(lasso)

#=====================================================================================
# 预测模型 Boosting
from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X_train_scaled, y_train)
plotModelResults(xgb, X_train=X_train_scaled,X_test=X_test_scaled, plot_intervals=True, plot_anomalies=True)