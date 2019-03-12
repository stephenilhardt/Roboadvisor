import numpy as np
import pandas as pd
import time

import pandas_datareader.data as pdr
from sqlalchemy import create_engine
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

	yahoo_df = yahoo_df.drop('Ticker', axis=1)
	yahoo_df = yahoo_df.drop('Adj Close', axis=1)
	yahoo_df.to_csv(filepath + '{}.csv'.format(ticker))

	return -1

def sql_to_csv(config, filepath):
	engine = create_engine('postgres://{username}@{host}:{port}/{database}')
	query = 'SELECT * FROM prices JOIN dividends ON Ticker, Date LEFT'
	df = pd.read_sql(query, engine)

	tickers = df['Ticker'].unique()

	for ticker in tickers:
		ticker_df = df[df['Ticker'] == ticker]
		ticker_csv = pd.DataFrame(ticker_df[['Date','Open','High','Low','Close','Volume','Dividend']])
		ticker_csv['Split'] = 1

		ticker_csv.to_csv(filepath + '{}.csv'.format(ticker))

if __name__ == '__main__':
	get_yahoo_prices_all_tickers()