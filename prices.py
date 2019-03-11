import numpy as np
import pandas as pd
import time

import pandas_datareader.data as pdr
import fix_yahoo_finance as yf

yf.pdr_override()

def get_yahoo_prices_one_ticker(ticker):
	yahoo_df = pdr.get_data_yahoo(ticker, actions=True)
	yahoo_df['Ticker'] = ticker

	return yahoo_df

def get_yahoo_prices_all_tickers(tickers):	
	yahoo_prices = pd.DataFrame()

	for ticker in tickers:
		try:
			yahoo_df = get_yahoo_prices_one_ticker(ticker)
			yahoo_prices.append(yahoo_df)
		except:
			pass
		time.sleep(5 * np.random.random())

	return yahoo_prices

def price_to_csv(filepath, ticker):
	yahoo_df = get_yahoo_prices_one_ticker(ticker)

	yahoo_df = yahoo_df.drop('Ticker')
	yahoo_df = yahoo_df.drop('Adj Close')
	yahoo_df.to_csv(filepath + '{}.csv'.format(ticker))

	time.sleep(2)
	return -1

if __name__ == '__main__':
	get_yahoo_prices_all_tickers()