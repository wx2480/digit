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

warnings.filterwarnings('ignore')

# 模型相关的package
from xgboost import XGBClassifier, plot_importance, XGBRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from gplearn.genetic import SymbolicTransformer


# 读取未经过处理的数据，需要判断行权日以及空缺值，还需要观察形态
def get_snapshot():
    snapshot = pd.read_csv('C:/wxn/csv/market.csv', index_col=None, header=None)
    snapshot.columns = ['kind', 'datetime', 'open', 'high', 'low', 'close', 'vol']
    snapshot.drop_duplicates(inplace=True)
    snapshot.reset_index(drop=True, inplace=True)
    return snapshot


# 逐笔成交数据
def get_tick():
    tick = pd.read_csv('C:/wxn/csv/depth.csv', index_col=None, header=None)
    tick.columns = ['amount', 'price', 'datetime', 'type', 'kind']
    tick.drop_duplicates(inplace=True)
    tick.reset_index(drop=True, inplace=True)
    return tick


# 读取现货数据
def get_spot():
    spot = pd.read_csv('C:/wxn/csv/spot.csv', index_col=None, header=None)
    spot.columns = ['datetime', 'kind', 'open', 'high', 'low', 'close', 'vol']
    spot.drop_duplicates(inplace=True)
    spot.reset_index(drop=True, inplace=True)

    spot.datetime = spot.datetime.apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:8] + x[8:] + ':00')
    spot.set_index('datetime', inplace=True)
    return spot


# 期权，填充时间轴
def get_digit_opt_continue(digit_name, snapshot):
    y = snapshot[snapshot['kind'] == digit_name]
    y = y.groupby('datetime').tail(1).set_index('datetime')

    timeline = pd.date_range(y.index[0], y.index[-1], freq='1min')
    timeline = list(map(str, timeline))

    _timeline = []
    for i in timeline:
        if i not in y.index:
            _timeline.append(i)

    _y = pd.DataFrame(
        [[digit_name, np.nan, np.nan, np.nan, np.nan, np.nan]] * len(_timeline),
        index=_timeline,
        columns=y.columns,
    )

    y = pd.concat([_y, y]).sort_index()
    return y


# 获取一个种类的现货数据
def get_digit(name, spot):
    return spot[spot.kind == name]


# 判断行权日（不精确）
def get_strike_day(opt):
    st = opt.apply(lambda x: (np.isnan(x.close)) and (x.name[-8:-6] == '16'), axis=1)
    st = st[st]
    st.index = st.index.map(lambda x: x[:10])
    st = st.groupby(st.index).sum()

    ans = st[st == 60].index.tolist()
    return ans


# 计算因子
def cal_factors(y, c, p):
    long_windows = 60
    medium_windows = 35
    short_windows = 10
    factors_num = 42

    fac0 = np.log(y.close).diff(long_windows)
    fac1 = np.log(y.close).diff(short_windows)
    fac2 = np.log(y.close).diff(long_windows).rolling(long_windows).std()
    fac3 = np.log(y.close).diff(short_windows).rolling(short_windows).std()
    fac4 = ((y.high - y.low) / (y.close + y.open)).rolling(long_windows).mean()
    fac5 = ((y.high - y.low) / (y.close + y.open)).rolling(short_windows).mean()
    fac6 = y.vol.rolling(long_windows).sum()
    fac7 = y.vol.rolling(short_windows).sum()

    fac8 = np.log(c.close).diff(long_windows)
    fac9 = np.log(c.close).diff(short_windows)
    fac10 = np.log(c.close).diff().rolling(long_windows).std()
    fac11 = np.log(c.close).diff().rolling(short_windows).std()
    fac12 = ((c.high - c.low) / (c.close + c.open)).rolling(long_windows).mean()
    fac13 = ((c.high - c.low) / (c.close + c.open)).rolling(short_windows).mean()
    fac14 = c.vol.rolling(long_windows).sum()
    fac15 = c.vol.rolling(short_windows).sum()

    fac16 = np.log(p.close).diff(long_windows)
    fac17 = np.log(p.close).diff(short_windows)
    fac18 = np.log(p.close).diff().rolling(long_windows).std()
    fac19 = np.log(p.close).diff().rolling(short_windows).std()
    fac20 = ((p.high - p.low) / (p.close + p.open)).rolling(long_windows).mean()
    fac21 = ((p.high - p.low) / (p.close + p.open)).rolling(short_windows).mean()
    fac22 = p.vol.rolling(long_windows).sum()
    fac23 = p.vol.rolling(short_windows).sum()

    fac24 = (c.vol - p.vol) / (c.vol + p.vol + 1).rolling(long_windows).mean()
    fac25 = (c.close.diff()) / (y.close.diff() + 0.0001).rolling(long_windows).mean()
    fac26 = (p.close.diff()) / (y.close.diff() + 0.0001).rolling(long_windows).mean()

    fac27 = np.log(y.close).diff(medium_windows)
    fac28 = np.log(y.close).diff(medium_windows).rolling(medium_windows).std()
    fac29 = ((y.high - y.low) / (y.close + y.open)).rolling(medium_windows).mean()
    fac30 = np.log(c.close).diff(medium_windows)
    fac31 = np.log(p.close).diff(medium_windows)
    fac32 = (c.close.diff()) / (y.close.diff() + 0.0001).rolling(short_windows).mean()
    fac33 = (p.close.diff()) / (y.close.diff() + 0.0001).rolling(short_windows).mean()
    fac34 = (c.close.diff().diff()) / (y.close.diff().diff() + 0.0001).rolling(short_windows).mean()
    fac35 = (p.close.diff().diff()) / (y.close.diff().diff() + 0.0001).rolling(short_windows).mean()
    fac36 = (c.close.diff().diff()) / ((y.close.diff().diff()).diff() + 0.0001)
    fac37 = (p.close.diff().diff()) / ((y.close.diff().diff()).diff() + 0.0001)
    fac38 = c.close.diff() / (y.close.rolling(medium_windows).std().diff() + 0.0001)
    fac39 = p.close.diff() / (y.close.rolling(medium_windows).std().diff() + 0.0001)
    fac40 = fac38.rolling(medium_windows).mean()
    fac41 = fac39.rolling(medium_windows).mean()

    fac42 = y.close.rolling(15).mean()  # MA
    fac43 = y.close.rolling(15).apply(lambda x: x[x > 0].sum() / np.sum(np.abs(x)))  # RSI
    # fac44 =

    allfac = []
    for i in range(factors_num):
        '''
        locals()['fac{}'.format(i)] = (
            locals()['fac{}'.format(i)]).rolling(30).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std())
        '''
        allfac.append(locals()['fac{}'.format(i)])
    allfac = pd.concat(allfac, axis=1)

    columns = ['fac{}'.format(i) for i in range(factors_num)]

    allfac.columns = columns
    return allfac

    # fac0 = np.log(_y.close).diff(long_windows)
    # fac1 = np.log(_y.close).diff(short_windows)
    # fac2 = np.log(_y.close).diff(long_windows).rolling(long_windows).std()
    # fac3 = np.log(_y.close).diff(short_windows).rolling(short_windows).std()
    # fac4 = ((_y.high - _y.low) / (_y.close + _y.open)).rolling(long_windows).mean()
    # fac5 = ((_y.high - _y.low) / (_y.close + _y.open)).rolling(short_windows).mean()
    # fac6 = _y.vol.rolling(long_windows).sum()
    # fac7 = _y.vol.rolling(short_windows).sum()

    # fac8 = np.log(_c0.close).diff(long_windows)
    # fac9 = np.log(_c0.close).diff(short_windows)
    # fac10 = np.log(_c0.close).diff().rolling(long_windows).std()
    # fac11 = np.log(_c0.close).diff().rolling(short_windows).std()
    # fac12 = ((_c0.high - _c0.low) / (_c0.close + _c0.open)).rolling(long_windows).mean()
    # fac13 = ((_c0.high - _c0.low) / (_c0.close + _c0.open)).rolling(short_windows).mean()
    # fac14 = _c0.vol.rolling(long_windows).sum()
    # fac15 = _c0.vol.rolling(short_windows).sum()

    # fac16 = np.log(_p0.close).diff(long_windows)
    # fac17 = np.log(_p0.close).diff(short_windows)
    # fac18 = np.log(_p0.close).diff().rolling(long_windows).std()
    # fac19 = np.log(_p0.close).diff().rolling(short_windows).std()
    # fac20 = ((_p0.high - _p0.low) / (_p0.close + _p0.open)).rolling(long_windows).mean()
    # fac21 = ((_p0.high - _p0.low) / (_p0.close + _p0.open)).rolling(short_windows).mean()
    # fac22 = _p0.vol.rolling(long_windows).sum()
    # fac23 = _p0.vol.rolling(short_windows).sum()

    # fac24 = (_c0.vol - _p0.vol) / (_c0.vol+_p0.vol+1).rolling(long_windows).mean()
    # fac25 = (_c0.close.diff()) / (_y.close.diff() + 0.0001).rolling(long_windows).mean()
    # fac26 = (_p0.close.diff()) / (_y.close.diff() + 0.0001).rolling(long_windows).mean()

    # fac27 = np.log(_y.close).diff(medium_windows)
    # fac28 = np.log(_y.close).diff(medium_windows).rolling(medium_windows).std()
    # fac29 = ((_y.high - _y.low) / (_y.close + _y.open)).rolling(medium_windows).mean()
    # fac30 = np.log(_c0.close).diff(medium_windows)
    # fac31 = np.log(_p0.close).diff(medium_windows)
    # fac32 = (_c0.close.diff()) / (_y.close.diff() + 0.0001).rolling(short_windows).mean()
    # fac33 = (_p0.close.diff()) / (_y.close.diff() + 0.0001).rolling(short_windows).mean()
    # fac34 = (_c0.close.diff().diff()) / (_y.close.diff().diff() + 0.0001).rolling(short_windows).mean()
    # fac35 = (_p0.close.diff().diff()) / (_y.close.diff().diff() + 0.0001).rolling(short_windows).mean()
    # fac36 = (_c0.close.diff().diff()) / ((_y.close.diff().diff()).diff() + 0.0001)
    # fac37 = (_p0.close.diff().diff()) / ((_y.close.diff().diff()).diff() + 0.0001)
    # fac38 = _c0.close.diff()/(_y.close.rolling(medium_windows).std().diff() + 0.0001)
    # fac39 = _p0.close.diff()/(_y.close.rolling(medium_windows).std().diff() + 0.0001)
    # fac40 = fac38.rolling(medium_windows).mean()
    # fac41 = fac39.rolling(medium_windows).mean()

    #     for i in range(factors_num):
    #         locals()['fac{}'.format(i)] = locals()['fac{}'.format(i)].rolling(10).apply(lambda x:((x-x.mean())/x.std()).iloc[-1])
    # columns = ['fac{}'.format(i) for i in range(factors_num)]

    # factors = []
    # for _i in range(factors_num):
    # factors.append(locals()['fac{}'.format(str(_i))])
    # factors = pd.concat(factors,axis = 1)
    # factors.columns = columns
