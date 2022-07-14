# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.10.1 64-bit
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import quantstats_custom as qs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
def vdir(obj):
    return [x for x in dir(obj) if not x.startswith('_')]
# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# %%
# fetch the daily returns for a stock
tickerName = 'BTC-USD'
stock = qs.utils.download_returns(tickerName)

# %%
print('sharpe={}'.format(stock.sharpe()))
print('return={}'.format((stock+1).prod()-1))
def no_signal(returns=stock,rolling_period=30):
    return returns.rolling(rolling_period).apply(lambda x: 1)

rolling_period = 21*6
signals={'momentum':{'function':qs.stats.momentum},'momentum_simple':{'function':qs.stats.momentum_simple}\
    ,'rolling_sharpe':{'function':qs.stats.rolling_sharpe}\
    ,'hodl':{'function':no_signal}\
        }
for name,signal in signals.items():
    signal['metric'] = signal['function'](returns=stock,rolling_period=rolling_period)
    print('{:16s} mean={}'.format(name,signal['metric'].mean()))
    plt.plot(signal['metric'],label=name)
plt.legend(prop={'family' : 'monospace'})
plt.title('Metric')
plt.show()

shift_amount=1
min_clip=0
max_clip=2
add_amount=0
divide_amount=1
start_date = pd.to_datetime('2017-01-01')
stock_series = stock.loc[stock.index>=start_date]
label = '{:16s} cagr={:6.2%} sharpe={:5.3f} vol={:6.2%} sortino={:5.3f}'

returns=stock_series
metrics=stock_series
for name,signal in signals.items():
    signal['signal'] = signal['metric'].shift(shift_amount)
    signal['signal'] = np.clip(signal['signal'],min_clip,max_clip)
    signal['signal'] = (signal['signal']+add_amount)/divide_amount
    # signal['signal'] = signal['signal'].loc[stock.index>=start_date]
    signal['signal'] = signal['signal']*signals['momentum']['signal'].mean()/signal['signal'].mean()
    print('{:16s} signal mean={}'.format(name,signal['signal'].mean()))
    signal['return'] = stock_series*(signal['signal'].loc[stock.index>=start_date])
    signal['return'] = signal['return']*0.1/qs.stats.volatility(signal['return']) # normalize to 10% volatility
    returns=pd.concat([returns,signal['return']],axis=1)
    metrics=pd.concat([metrics,signal['metric']],axis=1)
display(returns.head(2))
display(metrics.tail(2))

# add a new signal
signals['three_signals']={'return':signals['hodl']['metric'].loc[stock.index>=start_date]-1}
# display(signals['three_signals'])
for name,signal in signals.items():
    if name!='hodl':
        signals['three_signals']['return']=signals['three_signals']['return']+signal['return']/len(signals.items())
signals['three_signals']['return']=signals['three_signals']['return']*0.1/qs.stats.volatility(signals['three_signals']['return'])
for name,signal in signals.items():
    plt.plot(signal['return'],label=name)

plt.figure(figsize=(12,8))
for name,signal in signals.items():
    labelText = label.format(name,qs.stats.cagr(signal['return']),qs.stats.sharpe(signal['return']),qs.stats.volatility(signal['return']),qs.stats.sortino(signal['return']))
    print(labelText)
    plt.plot(qs.stats.compsum(signal['return']),label=labelText)
plt.legend(prop={'family' : 'monospace'})
plt.title('{} momentum strategies since {:%Y-%b-%d}'.format(tickerName,start_date))
plt.show()

# %%
# tickerList = ['TFLO','EMBH','IBTA.L','IBTU.L','SHV','DFNM']
tickerList = ['TFLO','IBTA.L','IBTU.L','SHV','USFR','PVI','OPER']
# tickerList = ['VHT','VOO']

stocks = pd.DataFrame()
startDates = dict()
for i in tickerList:
    newStock = qs.utils.download_returns(i)
    stocks = pd.concat([stocks, newStock], axis=1)
    startDates[i]=newStock.index[0]

# %%
stocks.columns = tickerList
display(stocks.tail(3))
display(startDates)
display(startDates.keys())
maxStartDate = max(startDates.values())
display(maxStartDate)

# %%
print('utils')
print(vdir(qs.utils))
print('stats')
print(vdir(qs.stats))
print('plots')
print(vdir(qs.plots))
help(qs.stats.comp)
help(qs.utils.make_portfolio)

# %%
# portfolio = qs.utils.make_index(stocks)
n = stocks.shape[1]+1
# display(n)
portfolio = stocks.mean(axis=1)
fig, ax = plt.subplots(ncols=2, nrows=n,gridspec_kw = {'wspace':0, 'hspace':0.01}, figsize=(12,4*n))
fig.patch.set_facecolor('white')
for i, stock in enumerate(stocks.columns):
    ax[i,0].plot(qs.stats.compsum(stocks[stock]), label=stock)
    ax[i,0].legend()
    ax[i,0].set_title(stock)
    ax[i,0].set_xlabel('Date')
    ax[i,0].set_ylabel('Returns')
    ax[i,1].plot(qs.stats.to_drawdown_series(stocks[stock]), label=stock)
    ax[i,1].legend()
    ax[i,1].set_title(stock)
    ax[i,1].set_xlabel('Date')
    ax[i,1].set_ylabel('Drawdown')
    # ax[i].grid(visible=True,linestyle='--', linewidth='1', color='grey',which='both',axis='y')
ax[n-1,0].plot(qs.stats.compsum(portfolio), label='Portfolio')
ax[n-1,0].set_title('Portfolio')
ax[n-1,0].set_xlabel('Date')
ax[n-1,0].set_ylabel('Returns')
ax[n-1,1].plot(qs.stats.to_drawdown_series(portfolio), label='Portfolio')
ax[n-1,1].set_title('Portfolio')
ax[n-1,1].set_xlabel('Date')
ax[n-1,1].set_ylabel('Returns')
plt.show()
# qs.plots.snapshot(portfolio, title='Performance')

# %%
# portfolio = qs.utils.make_index(stocks)
n = stocks.shape[1]+1
# display(n)
labelFormat =  '{:12s} return={:8.2%} vol={:6.2%} sharpe={:6.2f} sortino={:6.2f} maxDD={:6.2%}'
portfolio = stocks.mean(axis=1)
fig, ax = plt.subplots(ncols=2, nrows=1,gridspec_kw = {'wspace':0.05, 'hspace':0.01}, figsize=(20,8))
fig.patch.set_facecolor('white')
for i, stock in enumerate(stocks.columns):
    relevantData = stocks[stock].loc[stocks[stock].index>=maxStartDate]
    relevantLabel = labelFormat.format(stock,qs.stats.cagr(relevantData),qs.stats.volatility(relevantData),qs.stats.sharpe(relevantData),qs.stats.sortino(relevantData),qs.stats.max_drawdown(relevantData))
    ax[0].plot(qs.stats.compsum(relevantData), label=relevantLabel)
    ax[0].set_title(stock)
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Percent Return')
    ax[1].plot(qs.stats.to_drawdown_series(relevantData), label=relevantLabel)
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Percentage Drawdown')
relevantData = portfolio.loc[portfolio.index>=maxStartDate]
relevantLabel = labelFormat.format('Portfolio',qs.stats.cagr(relevantData),qs.stats.volatility(relevantData),qs.stats.sharpe(relevantData),qs.stats.sortino(relevantData),qs.stats.max_drawdown(relevantData))
ax[0].plot(qs.stats.compsum(relevantData), label=relevantLabel)
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Returns')
ax[0].legend()
ax[0].set_title('Cumulative Return')
ax[1].plot(qs.stats.to_drawdown_series(relevantData), label=relevantLabel)
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Returns')
ax[1].legend()
ax[1].set_title('Drawdown')
plt.show()

# %%
# additional detail
# qs.stats.compsum(stocks.loc[stocks.index>=maxStartDate])
# for stock in tickerList:
#     display(qs.reports.full(returns=stocks.loc[stocks[stock].index>=maxStartDate,stock]))

# %%
# report = qs.reports.metrics(mode='full', returns=stock)

# %%
# qs.reports.plots(mode='full', returns=stock) # shows basic/full metrics

# %%
# qs.reports.basic(returns=stock) # shows basic metrics and plots

# %%
qs.reports.full(returns=stocks.loc[:,'OPER']) # shows full metrics and plots

# %%
# qs.reports.html(returns=stock) # generates a complete report as html
