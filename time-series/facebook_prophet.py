# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings('ignore')

from zipfile import ZipFile
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""代码功能：预测meidium网站每天的发帖量"""

"""Prophet安装"""
# pip install fbprophet
# or
# conda install -c conda-forge fbprophet

# ==========================================================================
"""查看数据集"""


def tarfile(filepath):
    """打开压缩文件"""
    myzip = ZipFile(filepath)
    f = myzip.open('medium_posts.csv')
    df = pd.read_csv(f)
    # print(df)
    f.close()
    myzip.close()
    return df


# --------------------------------------------------------------------------
filepath = r'.\test\medium_posts.csv.zip'
df = tarfile(filepath)

# --------------------------------------------------------------------------
# 删除缺失值和重复值
df = df[['published', 'url']].dropna().drop_duplicates()

# 将时间字符串转换为datatime
df['published'] = pd.to_datetime(df['published'].str[:-1])

# 按时间排序,查看数据
df.sort_values(by=['published']).head(n=3)

"""《Medium》的公映日期是2012年8月15日。但是，从上面的数据可以看到，
至少有几行具有更早的发布日期。它们以某种方式出现在数据集中，
但它们不是合法的。将只保留2012年8月15日到2017年6月25日之间的时间序列
"""
df = df[(df['published'] > '2012-08-15') &
        (df['published'] < '2017-06-26')].sort_values(by=['published'])

df.head(n=3)

df.tail(n=3)

# 计算每个时间点的发帖数量 posts
aggr_df = df.groupby('published')[['url']].count()
aggr_df.columns = ['posts']
aggr_df.head(n=3)

"""为了解决这个问题，我们需要根据日期大小的“bin”来聚合帖子计数。
在时间序列分析中，这个过程被称为重采样。如果我们降低数据的采样率，这通常被称为向下采样。"""

"""pandas 内置函数，将时间索引重新取样到1天的 bin 中"""
daily_df = aggr_df.resample('D').apply(sum)
daily_df.head(3)


# ==========================================================================
"""数据可视化"""

"""与往常一样，查看数据的图形表示可能很有帮助和指导意义。
我们将为整个时间范围创建一个时间序列图。
在如此长的一段时间内显示数据可以提供有关季节性和明显异常偏差的线索"""


#
# init_notebook_mode(connected=True)  # 绘图初始化


# def plotly_df(df, title=''):
#     """对列数据可视化"""
#     common_kw = dict(x=df.index, mode='lines')
#     data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
#     layout = dict(title=title)
#     fig = dict(data=data, layout=layout)
#     iplot(fig, show_link=False)
#
# # 绘制数据
# plotly_df(daily_df, title='Posts on Medium (daily)')


# 绘制数据

def plot_df(df, name):
    df.plot()
    plt.grid(True)
    plt.title('Posts on Medium (%s)' % name)


plot_df(daily_df, 'daily')

"""放大数据并未推断出任何有意义的东西,除了明显的上升和加速趋势。
因为通常高频数据是很难分析的。"""

"""为了减少噪音，将邮件数量重新采样到每周。
除了重采样，也可以用移动平均平滑和指数平滑等降噪技术。"""

weekly_df = daily_df.resample('W').sum()

plot_df(weekly_df, 'weekly')

"""这张图比较适合分析。周采样的数据量要比日采样的少，
这样能够快速进入不同时间轴，便于理解数据和找到可能的趋势、周期和不规则效果的线索。
如连续几年放大圣诞节时间点对应的数据，会发现节日对人类行为（发帖）的影响。
现在要去除2015年前的数据，因为这部分数据对于预测2017年的发帖量的价值不大。同时，15年前每天的发帖量很少，
这会增加预测的噪声，因为模型将被迫适应这些不正常的历史数据以及与这几年相关的数据与指标。
"""

# 去除15年前的数据
daily_df = daily_df.loc[daily_df.index >= '2015-01-01']
daily_df.head(3)

"""从上面的可视化分析可以看出，数据集是非平稳的，并有明显的增长趋势。
还显示了每周和每年的季节性以及每年一些不正常的天。
"""

"""Prophet API 在 sklearn 中，首先创建模型，然后fit，最后预测。拟合方法的输入是dataframe，有两列：
ds: (datestamp) must be of type date or datetime.
y: is a numeric value we want to predict.
"""

from fbprophet import Prophet

import logging

logging.getLogger().setLevel(logging.ERROR)

# 将数据转换成prophet要求的格式

df = daily_df.reset_index()
df.columns = ['ds', 'y']
df.tail(3)


"""库的作者建议根据至少几个月的历史数据（最好一年以上）做出预测。
在本例中，数据是好几年的。为了度量预测的标准，
将数据集换分为历史部分和预测部分，前者位于时间的前端，后者位于时间轴的后端。
选用最后一个月数据作为预测目标。"""

prediction_size = 30
train_df = df[:-prediction_size]
train_df.tail(3)

#             ds    y
# 874 2017-05-24  375
# 875 2017-05-25  298
# 876 2017-05-26  269

"""首先创建prophet对象，这里可以将模型参数传到构造器中。
本文用默认的参数。然后用fit方法训练模型。"""

m = Prophet()
m.fit(train_df);

"""用 Prophet.make_future_dataframe方法创建一个dataframe，它会包含历史上的所有日期以及未来30天的日期。"""

future = m.make_future_dataframe(periods=prediction_size)
future.tail(n=3)

#             ds
# 904 2017-06-23
# 905 2017-06-24
# 906 2017-06-25
"""预测时传入预测的日期。如果提供了历史日期，则对历史数据进行样本内拟合"""
forecast = m.predict(future)
forecast.tail()

#             ds       trend  ...  multiplicative_terms_upper        yhat
# 902 2017-06-21  276.555435  ...                         0.0  292.095427
# 903 2017-06-22  277.273732  ...                         0.0  288.927322
# 904 2017-06-23  277.992028  ...                         0.0  280.234641
# 905 2017-06-24  278.710324  ...                         0.0  243.644924
# 906 2017-06-25  279.428621  ...                         0.0  247.995791
# [5 rows x 19 columns]

"""在得到的预测datafrfame中，可以看到很多列，包括趋势和季节成分及其置信区间。yhat列是预测得到的数据。
Prophet包有自己的可视化工具，可以直接用于评估结果(Prophet.plot)"""

m.plot(forecast)

"""这张图提供的有用信息不多，但是可以从中看出该模型将很多数据点视为异常值。
可以尝试绘制趋势和季节成分(Prophet.plot_components),这个可能更有用。如果给模型提供假日和活动信息,
也可以显示在图中。"""

m.plot_components(forecast)

"""从趋势图中可以看出，模型在2016年年末的加速增长阶段拟合的较好。
从每周的季节性图表可以，周六和周日的发帖量比一周的其他天要少。
在每年的季节性图表中，圣诞节那天有一个显著的下降。"""

"""预测质量评估"""
"""接下来计算预测的最近30天的误差来评估算法的质量。
计算误差需要观测值和预测值。"""

print(', '.join(forecast.columns))
"""可以看出，除了历史数据，这个dataframe包含我们需要的所有信息。
这里定义一个函数，将原始数据集与预测对象连接起来。"""


def make_comparison_dataframe(historical, forecast):
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]. \
        join(historical.set_index('ds'))


cmp_df = make_comparison_dataframe(df, forecast)
cmp_df.tail(3)
#                   yhat  yhat_lower  yhat_upper    y
# ds
# 2017-06-23  280.234641  255.764140  306.103197  421
# 2017-06-24  243.644924  219.233201  267.107170  277
# 2017-06-25  247.995791  224.896550  272.822630  253

"""用MAPE和MAE测量来衡量我们的预测质量"""


def calculate_forecast_errors(df, prediction_size):
    df = df.copy()
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    prediction_part = df[-prediction_size:]
    error_mean = lambda error_name: np.mean(np.abs(prediction_part[error_name]))
    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}


for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print(err_name, err_value)

# MAPE 22.566119266921035
# MAE 69.62700966533414

"""
模型可视化，包括实际值、预测值和置信区间。
首先绘制较短时间内的数据，使数据点容易区分。
其次仅在预测的时间段显示模型性能。这里定义一个show_forecast函数。
"""


def show_forecast(cmp_df, num_predictions, num_values, title):
    plt.figure(figsize=(10, 4))
    plt.plot(cmp_df['y'][-num_values:], 'r', label='actual')
    plt.plot(cmp_df['yhat'][-num_predictions:], 'b', label='forecast')
    plt.plot(cmp_df['yhat_upper'][-num_predictions:], 'gray', alpha=0.6, linestyle='dashed', label='upper/lower bond')
    plt.plot(cmp_df['yhat_lower'][-num_predictions:], 'gray', alpha=0.6, linestyle='dashed')
    plt.legend(loc='upper left')
    plt.fill_between(x=cmp_df.index[-num_predictions:], y1=cmp_df['yhat_upper'][-num_predictions:],
                     y2=cmp_df['yhat_lower'][-num_predictions:], alpha=0.5, color='lightgrey')
    plt.title(title)
    plt.grid(True)


show_forecast(cmp_df, prediction_size, 100, 'New posts on Medium')

"""乍看，模型对平均值的预测似乎是合理的。高的MAPE说明模型未能捕捉到弱季节性的峰间振幅增加。
从上图可以看出，许多实际值位于置信区间外。Prophet可能不适合具有不稳定方差的时间序列，至少使用默认设置时是这样。
下面尝试对数据进行转换来解决这个问题。"""

"""Box-Cox转换"""
"""目前为止，已经使用了默认设置的Prophet和原始数据。这里我们不考虑模型参数，但可以对原始数据进行转换。
我们将把Box-Cox转换应用到原始时序中。
这是一个单调数据转换，可以用来稳定方差。我们用单参数Box-Cox转换，定义如下："""

# 对转换取反对数
"""反对数，是指如果正数n的对数为b，则称n为b的反对数。如果logt（n）=b，则称n为b的反对数，记作n＝t^b。
反对数就是由已知对数b去求出相应的真数n。"""


def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp((np.log(lambda_ * y + 1)) / lambda_)


# 数据设置索引
train_df2 = train_df.copy().set_index('ds')

"""用Scipy.stats.boxcox 函数对数据进行转换，有两个返回值：
第一个是转换后的序列，第二个是通过最大对数似然法找到的最优 lambda。"""
train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])
train_df2.reset_index(inplace=True)

# 重新构建Prophet模型
m2 = Prophet()
m2.fit(train_df2)
future2 = m2.make_future_dataframe(periods=prediction_size)
forecast2 = m2.predict(future2)

# 根据 lambda 和反函数得到真实值
for column in ['yhat', 'yhat_lower', 'yhat_upper']:
    forecast2[column] = inverse_boxcox(forecast2[column], lambda_prophet)

# 将历史数据与预测数据变成一个dataframe
cmp_df2 = make_comparison_dataframe(df, forecast2)

# 计算误差
for err_name, err_value in calculate_forecast_errors(cmp_df2, prediction_size).items():
    print(err_name, err_value)

# MAPE 11.61975850181049
# MAE 39.19352456749703
"""可以看出，模型的质量在提高。最后，将以前的性能与最新的结果并排绘制在一起。"""
show_forecast(cmp_df, prediction_size, 100, 'No transformations')
show_forecast(cmp_df2, prediction_size, 100, 'Box-Cox transformation')

"""可以看到，第二张图中对周变化的预测更接近于实际值"""
"""Prophet是一个专门针对商业时间序列的开源预测库。在实践中发现其预测不理想，但通过数据转换可以达到较好的效果。"""

