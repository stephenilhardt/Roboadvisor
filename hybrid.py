import numpy as np
import pandas as pd

from fbprophet import Prophet

from keras.layers import Dense, Activation, GRU, Dropout
from keras.models import Sequential

from sklearn.preprocessing import StandardScaler

macro = pd.read_csv('/home/ubuntu/csvs/macro_pivot.csv')
macro = macro.set_index('date')
macro.index = pd.to_datetime(macro.index)

etfs = pd.read_csv('/home/ubuntu/csvs/etf_pivot.csv')
etfs = etfs.set_index('date')
etfs.index = pd.to_datetime(etfs.index)

first_year = pd.date_range(start = '2014-03-17', end = '2015-03-16').tolist()
second_year = pd.date_range(start = '2015-03-17', end = '2016-03-15').tolist()
third_year = pd.date_range(start = '2016-03-`17', end = '2017-03-16').tolist()
fourth_year = pd.date_range(start = '2017-03-17', end = '2018-03-16').tolist()
fifth_year = pd.date_range(start = '2018-03-17', end = '2019-03-15').tolist()

def train_gru(etf):
	etf_df = pd.DataFrame(etfs[etf])
	first_year_etf = etf_df[etf_df.index.isin(first_year)]
	first_year_etf = first_year_etf.reset_index()
	first_year_etf.columns = ['ds', 'y']

	arima_model = Prophet()
	arima_model.fit(first_year_etf)

	future = arima_model.make_future_dataframe(periods = len(second_year))
	forecast = arima_model.predict(future)

	arima_predictions = forecast[forecast['ds'].isin(second_year)]['yhat']
	second_year_prices = etf_df[etf_df.index.isin(second_year)][etf]
	residuals = np.array(second_year_prices) - np.array(arima_predictions)

	macro_frame = macro[macro.index.isin(third_year)]
	macro_frame['residuals'] = residuals
	macro_array = macro_frame.values
	X = np.reshape(macro_array, (macro_array.shape[0], macro_array.shape[1], 1))

	fourth_year_prices = etf_df[etf_df.index.isin(fourth_year)]
	y = np.array(fourth_year_prices)

	gru_model = Sequential()

	gru_model.add(GRU(units=100, input_shape=(28780,1), return_sequences=True))
	gru_model.add(Dropout(0.2))
	gru_model.add(GRU(units=50, return_sequences=True))
	gru_model.add(Dropout(0.2))
	gru_model.add(GRU(units=25, return_sequences=True))
	gru_model.add(Dropout(0.2))
	gru_model.add(GRU(units=10, return_sequences=False))
	gru_model.add(Dropout(0.2))
	gru_model.add(Dense(1))
	gru_model.add(Activation('linear'))

	gru_model.compile(loss='mse', optimizer='rmsprop')
	gru_model.fit(x=X, y=y, batch_size=365, epochs=10, validate=0.25)

	return gru_model

def predict(etf, gru_model):
	etf_df = pd.DataFrame(etfs[etf])
	second_year_etf = etf_df[etf_df.index.isin(second_year)]
	second_year_etf = second_year_etf.reset_index()
	second_year_etf.columns = ['ds', 'y']

	arima_model = Prophet()
	arima_model.fit(second_year_etf)

	future = arima_model.make_future_dataframe(periods = len(third_year))
	forecast = arima_model.predict(future)

	arima_predictions = forecast[forecast['ds'].isin(second_year)]['yhat']
	third_year_prices = etf_df[etf_df.index.isin(third_year)][etf]
	residuals = np.array(third_year_prices) - np.array(arima_predictions)

	macro_frame = macro[macro.index.isin(fourth_year)]
	macro_frame['residuals'] = residuals
	macro_array = macro_frame.values
	X = np.reshape(macro_array, (macro_array.shape[0], macro_array.shape[1], 1))

	predictions = gru_model.predict(X)

	return predictions