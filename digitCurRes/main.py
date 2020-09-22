# 基础的package
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import os
import warnings
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# 模型相关的package
from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from gplearn.genetic import SymbolicTransformer

# 自定义模块
from utils.utils import get_snapshot
from utils.utils import get_tick
from utils.utils import get_spot
from utils.utils import get_digit_opt_continue
from utils.utils import get_strike_day
from utils.utils import get_digit, cal_factors


# 判断方向
def f(x):
    if x > 0:
        return 1
    else:
        return -1


f = np.vectorize(f)

plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.axisbelow'] = True
colors = ['#44A948', '#137CB1', '#EACC80', '#A8D7E2', '#E58061']

snapshot = get_snapshot()
tick = get_tick()
spot = get_spot()

digit_name_c0 = 'market0_486_1'
digit_name_p0 = 'market0_487_1'
digit_name_c1 = 'market0_516_1'
digit_name_p1 = 'market0_517_1'

# digit_name_c0 = 'market0_514_1'
# digit_name_p0 = 'market0_515_1'
# digit_name_c1 = 'market0_551_1'
# digit_name_p1 = 'market0_551_1'

c0 = get_digit_opt_continue(digit_name_c0, snapshot)
p0 = get_digit_opt_continue(digit_name_p0, snapshot)
c1 = get_digit_opt_continue(digit_name_c1, snapshot)
p1 = get_digit_opt_continue(digit_name_p1, snapshot)
y = get_digit('BTCP', spot)

# 计算因子
# ret = -np.log(y.close).diff(-10)
end = ' 16:00:00'
start = ' 17:00:00'

ret_time = 1

timepoint = get_strike_day(c0)

for i in range(3):
    asdf = i
    if (i == 2) | (i == 0):
        pass
        # continue

    if i == 2:
        _y = y.loc[timepoint[i] + start:].fillna(method='ffill')
        _c0 = c0.loc[timepoint[i] + start:].fillna(method='ffill')
        _p0 = p0.loc[timepoint[i] + start:].fillna(method='ffill')
    else:
        _y = y.loc[timepoint[i] + start:timepoint[i + 1] + end].fillna(method='ffill')
        _c0 = c0.loc[timepoint[i] + start:timepoint[i + 1] + end].fillna(method='ffill')
        _p0 = p0.loc[timepoint[i] + start:timepoint[i + 1] + end].fillna(method='ffill')

    position = []
    ret = ((_c0.close.shift(-ret_time) / (_c0.close)) - 1)

    factors = cal_factors(_y, _c0, _p0)
    Y1 = f(np.nan_to_num(ret))
    Y2 = f(np.nan_to_num(ret))
    X1 = np.nan_to_num(factors.values)
    X2 = np.nan_to_num(factors.values)

    X_train1, Y_train1 = X1[:9999, :], Y1[:9999]
    X_train2, Y_train2 = X2[:9999, :], Y2[:9999]
    model1 = XGBClassifier(n_jobs=7)
    model2 = XGBClassifier(n_jobs=7)
    r1 = model1.fit(X_train1, Y_train1)
    r2 = model2.fit(X_train2, Y_train2)

    print('model1: ', r1.score(X1[10000:, :], Y1[10000:]))
    print('model2: ', r2.score(X2[10000:, :], Y2[10000:]))

    for _i in range(10000, _y.shape[0] - 120):
        X_test, Y_test = X1[_i:_i + 1, :], Y1[_i:_i + 1]
        prey1 = r1.predict(X_test)
        prey2 = r2.predict(X_test)
        position.append(float(prey2[0]))

        if _i % 5000 == 1450:
            gc.collect()
        if _i % 1000 == 450:
            print(_i - 1450, '/', _y.shape[0] - 120 - 1450)
        '''
        if _i % 14400 == 0:
            X_train1, Y_train1 =  X1[:_i,:], Y1[:_i]
            X_train2, Y_train2 = X2[:_i,:], Y2[:_i]
            model1 = XGBClassifier(n_jobs = 7)
            model2 = XGBClassifier(n_jobs = 7)
            r1 = model1.fit(X_train1, Y_train1)
            r2 = model2.fit(X_train2, Y_train2)
        '''
#         print(_i-1450, '/', _y.shape[0]-1440, ' ', r.score(X_train,Y_train),' ',Y_test[0], ' ', prey[0]

    pos = np.array(position)
    a = ret[10000:10000 + len(pos)]

    net = [[1 / ret_time] * ret_time]
    for i in range(0, len(pos) // ret_time):
        n = []
        for j in range(ret_time):
            _ = net[-1][j] + pos[ret_time * i + j] * a[ret_time * i + j] / ret_time
            if (i != 0) & (pos[ret_time * i + j] != pos[ret_time * i + j - ret_time]):
                _ = _ - 0.001 / ret_time
            n.append(_)
        net.append(n)

#     np.save('{}.npy'.format(asdf), pos)

    fig, ax = plt.subplots()
    net_worth = pd.DataFrame(np.array(net))
    if ret_time == 1:
        net_worth.index = _y.index[10000:10000 + len(net_worth)]
    ax.plot(net_worth.sum(axis=1), color=colors[4])
    for i, j in enumerate(ax.get_xticklabels()):
        if i % 7200 == 0:
            j.set_visible(True)
            j.set_rotation(10)
        else:
            j.set_visible(False)
    for i, j in enumerate(ax.get_xticklines()):
        if i % 7200 == 0:
            j.set_visible(True)
        else:
            j.set_visible(False)
    fig.savefig('{}.png'.format(asdf))
    plt.show(fig)