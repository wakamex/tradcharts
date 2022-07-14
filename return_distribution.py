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
sns.set() # make charts look cool
qs.extend_pandas() # extend pandas functionality with metrics, etc.

#helper functions
def vdir(obj):
    return [x for x in dir(obj) if not x.startswith('_')]
def gmean(df,axis=0):
    return np.exp(np.log((1+df/100).prod(axis=axis))/df.notna().sum(axis))-1


# %%
rois = pd.read_csv('rois.csv')
apys = pd.read_csv('apys.csv')

# %%
for name,df in {'ROIs':rois,'APYs':apys}.items():
    ax=plt.figure(figsize=(12,6))
    labelFormat = '{:10s} geomean={:8.2%}'
    cols = df.columns
    bins = 50 if name=='ROIs' else [-100,-99.1,0]+(10**np.arange(1,6,1)).tolist()
    counts1, bins = np.histogram(df.loc[:,cols[1]],bins=bins)
    if name == 'ROIs':
        xdata = bins[:-1]
    else:
        xdata = range(len(bins)-1)
        xdatalabels = ['{:.0f} exactly'.format(bins[i]) if bins[i+1]-bins[i]<1 else '{:.0f}-{:.0f}'.format(bins[i],bins[i+1]) for i in range(0,len(bins)-1)]
    plt.bar(xdata,counts1,label=labelFormat.format(cols[1],gmean(df.loc[:,cols[1]])),alpha=0.2,width=1,edgecolor='black')
    if name != 'ROIs':
        plt.gca().set_xticks(xdata)
        plt.gca().set_xticklabels(xdatalabels)
    counts2, bins = np.histogram(df.loc[:,cols[0]],bins=bins)
    plt.bar(xdata,counts2,label=labelFormat.format(cols[0],gmean(df.loc[:,cols[0]])),alpha=0.25,width=1,edgecolor='black')
    if name != 'ROIs':
        plt.gca().set_xticks(xdata)
        plt.gca().set_xticklabels(xdatalabels)
    res = pd.DataFrame(data=[counts1,counts2]).T
    res.columns = [name[:-1]+cols[1],name[:-1]+cols[0]]
    res['bucket_min'] = bins[:-1]
    res['bucket_max'] = bins[1:]
    res['total_freq'] = res.iloc[:,0]+res.iloc[:,1]
    display(res.loc[res.total_freq>0,:])
    plt.legend(prop={'family' : 'monospace'})
    plt.xlabel(name[:-1]+'(%)')
    plt.ylabel('# of trades')
plt.show()
