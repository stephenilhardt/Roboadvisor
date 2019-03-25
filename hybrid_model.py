import numpy as np
import pandas as pd

from fbprophet import Prophet

from keras.layers import Dense, Activation, GRU, Dropout
from keras.models import Sequential

from sklearn.preprocessing import StandardScaler

macro = pd.read_csv('/home/ubuntu/csvs/macro_pivot.csv')
macro = macro.set_index('date')
macro.index = pd.to_datetime(macro.index)

etfs = pd.read_csv('/home/ubuntu/csvs/etfs_pivot.csv')
etfs = etfs.set_index('date')
etfs.index = pd.to_datetime(etfs.index)

first_year = pd.date_range(start = '2014-03-17', end = '2015-03-16').tolist()
second_year = pd.date_range(start = '2015-03-17', end = '2016-03-15').tolist()
third_year = pd.date_range(start = '2016-03-17', end = '2017-03-16').tolist()
fourth_year = pd.date_range(start = '2017-03-17', end = '2018-03-16').tolist()
fifth_year = pd.date_range(start = '2018-03-17', end = '2019-03-15').tolist()

date_ranges = [first_year, second_year, third_year, fourth_year, fifth_year]

def build_gru_model():

	gru_model = Sequential()

	gru_model.add(GRU(units=100, input_shape=(28779,1), return_sequences=True))
	gru_model.add(Dropout(0.2))
	gru_model.add(GRU(units=50, return_sequences=False))
	gru_model.add(Dropout(0.2))
	gru_model.add(Dense(1))
	gru_model.add(Activation('linear'))

	gru_model.compile(loss='mse', optimizer='rmsprop')

	return gru_model

def train_gru_model(etf, gru_model):

	etf_df = pd.DataFrame(etfs[etf])

	for index, year in enumerate(date_ranges[:3]):

		#date_range = pd.date_range(start=date_ranges[index][0], end=date_ranges[index][-1])

		prices = etf_df[etf_df.index.isin(year)]
		prices = prices.reset_index()
		prices.columns = ['ds', 'y']

		arima_model = Prophet()
		arima_model.fit(prices)

		future = arima_model.make_future_dataframe(periods = (len(date_ranges[index+1])+1))
		forecast = arima_model.predict(future)

		arima_predictions = np.array(forecast[forecast['ds'].isin(date_ranges[index+1])]['yhat'])
		prices_array = np.array(etf_df[etf_df.index.isin(date_ranges[index+1])][etf])
		residuals = prices_array - arima_predictions

		scaler = StandardScaler()
		X_frame = macro[macro.index.isin(year)]
		X_unscaled = X_frame.values
		X_array = scaler.fit_transform(X_unscaled)
		X = np.reshape(X_array, (X_array.shape[0], X_array.shape[1], 1))
		y = np.array(residuals)

		gru_model.fit(x=X,y=y,epochs=1,batch_size=25,validation_split=0.2)

	return gru_model

def make_predictions(etf, gru_model):

	#date_range = pd.date_range(start=date_ranges[0][0], end=date_ranges[3][-1])

	etf_df = pd.DataFrame(etfs[etf])
	prices = etf_df[etf_df.index.isin(fourth_year)]
	prices = prices.reset_index()
	prices.columns = ['ds', 'y']

	arima_model = Prophet()
	arima_model.fit(prices)

	future = arima_model.make_future_dataframe(periods = len(fifth_year))
	forecast = arima_model.predict(future)
	predicted_prices = np.array(forecast[forecast['ds'].isin(fifth_year)]['yhat'])

	scaler = StandardScaler()
	X_frame = macro[macro.index.isin(fourth_year)]
	X_unscaled = X_frame.values
	X_array = scaler.fit_transform(X_unscaled)
	X = np.reshape(X_array, (X_array.shape[0], X_array.shape[1], 1))

	residuals_array = gru_model.predict(X)
	predicted_residuals = (residuals_array.reshape(residuals_array.shape[1], residuals_array.shape[0])[0]).tolist()
	predicted_residuals.pop()

	return_frame = pd.DataFrame(index=date_ranges[4])
	return_frame['arima'] = predicted_prices
	return_frame['residual'] = predicted_residuals
	return_frame['prediction'] = return_frame['arima'] + return_frame['residual']
	return_frame['actual'] = np.array(etf_df[etf_df.index.isin(fifth_year)])

	return return_frame


