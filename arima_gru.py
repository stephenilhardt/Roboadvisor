import numpy as np
import pandas as pd

from fbprophet import Prophet

from keras.layers import Dense, Activation, GRU, Dropout
from keras.models import Sequential

from sklearn.preprocessing import StandardScaler

date_ranges = [
	pd.date_range(start='2014-03-17', end='2014-06-16').tolist(),
	pd.date_range(start='2014-06-17', end='2014-09-16').tolist(),
	pd.date_range(start='2014-09-17', end='2014-12-16').tolist(),
	pd.date_range(start='2014-12-17', end='2015-03-16').tolist(),

	pd.date_range(start='2015-03-17', end='2015-06-16').tolist(),
	pd.date_range(start='2015-06-17', end='2015-09-16').tolist(),
	pd.date_range(start='2015-09-17', end='2015-12-16').tolist(),
	pd.date_range(start='2015-12-17', end='2016-03-16').tolist(),

	pd.date_range(start='2016-03-17', end='2016-06-16').tolist(),
	pd.date_range(start='2016-06-17', end='2016-09-16').tolist(),
	pd.date_range(start='2016-09-17', end='2016-12-16').tolist(),
	pd.date_range(start='2016-12-17', end='2017-03-16').tolist(),

	pd.date_range(start='2017-03-17', end='2017-06-16').tolist(),
	pd.date_range(start='2017-06-17', end='2017-09-16').tolist(),
	pd.date_range(start='2017-09-17', end='2017-12-16').tolist(),
	pd.date_range(start='2017-12-17', end='2018-03-16').tolist(),

	pd.date_range(start='2018-03-17', end='2018-06-16').tolist(),
	pd.date_range(start='2018-06-17', end='2018-09-16').tolist(),
	pd.date_range(start='2018-09-17', end='2018-12-16').tolist(),
	pd.date_range(start='2018-12-17', end='2019-03-15').tolist()
]

etfs = pd.read_csv('/home/ubuntu/csvs/etfs_pivot.csv', parse_dates=True, infer_datetime_format=True)
etfs = etfs.set_index('date')
etfs.index = pd.to_datetime(etfs.index)

macro = pd.read_csv('/home/ubuntu/csvs/macro_pivot.csv', parse_dates=True, infer_datetime_format=True)
macro = macro.set_index('date')
macro.index = pd.to_datetime(macro.index)

def arima_model(etf, date_range=date_ranges[0], date_ranges=date_ranges):

	etf_df = pd.DataFrame(etfs[etf])

	true_price = etf_df[etf_df.index.isin(date_range)]
	true_price = true_price.reset_index()
	true_price.columns = ['ds', 'y']

	arima_model = Prophet()
	arima_model.fit(true_price)

	return arima_model

def arima_residuals(etf, arima_model, date_range=date_ranges[0], date_ranges=date_ranges):

	dr_index = date_ranges.index(date_range)

	etf_df = pd.DataFrame(etfs[etf])

	future = arima_model.make_future_dataframe(periods = len(date_ranges[dr_index+1]))
	forecast = arima_model.predict(future)

	three_months_out = forecast[forecast['ds'].isin(date_ranges[dr_index+1])]
	predicted_prices = pd.DataFrame(three_months_out[['ds','yhat']])
	predicted_prices.columns = ['date','etf']
	predicted_prices = predicted_prices.set_index('date')

	return_frame = pd.DataFrame(index=predicted_prices.index)
	return_frame['prediction'] = predicted_prices.values
	return_frame['actual'] = etf_df[etf_df.index.isin(date_ranges[dr_index + 1])][etf].values

	return_frame['residual'] = return_frame['actual'] - return_frame['prediction']

	return return_frame

def gru_model(residuals, date_range=date_ranges[0], date_ranges=date_ranges):

	dr_index = date_ranges.index(date_range)

	scaler = StandardScaler()

	X_frame = macro[macro.index.isin(date_ranges[dr_index + 1])]
	X_unscaled = X_frame.values
	X_array = scaler.fit_transform(X_unscaled)
	X = np.reshape(X_array, (X_array.shape[0], X_array.shape[1], 1))

	y = np.array(residuals)

	gru_model = Sequential()

	gru_model.add(GRU(units=100, input_shape=(28779,1), return_sequences=True))
	gru_model.add(Dropout(0.2))
	gru_model.add(GRU(units=50, return_sequences=False))
	gru_model.add(Dropout(0.2))
	gru_model.add(Dense(1))
	gru_model.add(Activation('linear'))

	gru_model.compile(loss='mse', optimizer='rmsprop')
	gru_model.fit(x=X,y=y, batch_size=87, epochs = 5, validation_split=0.05)

	return gru_model

def final_predictions(etf, gru_model, arima_model, date_range=date_ranges[0], date_ranges=date_ranges):

	dr_index = date_ranges.index(date_range)

	new_future = arima_model.make_future_dataframe(periods=len(date_ranges[dr_index+1])+len(date_ranges[dr_index+2]))
	new_forecast = arima_model.predict(new_future)

	new_forecast_predictions = new_forecast[new_forecast['ds'].isin(date_ranges[dr_index+2])]['yhat'].values

	macro_predict = macro[macro.index.isin(date_ranges[dr_index+2])].values
	predictors = np.reshape(macro_predict, (macro_predict.shape[0], macro_predict.shape[1], 1))

	gru_residuals = gru_model.predict(predictors)

	return_frame = pd.DataFrame(index=date_ranges[dr_index+2])
	return_frame['arima_forecasts'] = new_forecast_predictions
	return_frame['gru_residuals'] = gru_residuals
	return_frame['predictions'] = return_frame['arima_forecasts'] + return_frame['gru_residuals']

	return return_frame

if __name__ == '__main__':
	main()