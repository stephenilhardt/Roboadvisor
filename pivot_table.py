import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
from dask.distributed import Client, progress

client = Client(n_workers=4, threads_per_worker=2)

fred = dd.read_csv('FRED_20190312.csv', header=None)

fred.columns = ['code','date','value']

fred.code = fred.code.astype('category').cat.as_known()

fred_pivot = fred.pivot_table(index='date', columns='code', values='value')

fred_pivot.to_csv('fred_pivot.csv')