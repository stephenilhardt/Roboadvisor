import numpy as np
import pandas as pd

import arima_gru
from fbprophet import Prophet

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

macro = pd.read_csv('/home/ubuntu/csvs/macro_pivot.csv', parse_dates=True, infer_datetime_format=True)
macro = macro.set_index('date')
macro.index = pd.to_datetime(macro.index)

etfs = pd.read_csv('/home/ubuntu/csvs/etfs_pivot.csv', parse_dates=True, infer_datetime_format=True)
etfs = etfs.set_index('date')
etfs.index = pd.to_datetime(etfs.index)

def build_gru(etf, gru_model, date_ranges=date_ranges):

	i = 1
	while i < 11:
		date_range = date_ranges[i]
		ts_date_range = pd.date_range(start='2014-03-17', end=date_range[-1]).tolist()

		arima_model = arima_gru.arima_model(etf, ts_date_range)
		arima = arima_gru.arima_residuals(etf, arima_model, date_range)

		X_frame = macro[macro.index.isin(date_ranges[i+1])]
		X_unscaled = X_frame.values

		scaler = StandardScaler()
		X_array = scaler.fit_transform(X_unscaled)
		X = np.reshape(X_array, (X_array.shape[0], X_array.shape[1], 1))
		y = np.array(arima['residual'])

		gru_model.fit(x=X, y=y, batch_size=87, epochs=5, validation_split=0.05)

		i += 1

	return gru_model

def build_arima(etf, arima_model, date_range, date_ranges=date_ranges):

	etf_df = pd.DataFrame(etfs[etf])
	etf_df = etf_df[etf_df.index.isin(date_range)]

	etf_df = etf_df.reset_index()
	etf_df.columns = ['ds','y']

	arima_model.fit(etf_df)

	return arima_model

def make_predictions(etf, gru_model, date_ranges=date_ranges):

	predictions_csv = pd.read_csv('/home/ubuntu/csvs/predictions.csv', parse_dates=True, infer_datetime_format=True)
	predictions_csv = predictions_csv.set_index('date')
	predictions_csv.index = pd.to_datetime(predictions_csv.index)

	residuals_csv = pd.read_csv('/home/ubuntu/csvs/residuals.csv', parse_dates=True, infer_datetime_format=True)
	residuals_csv = residuals_csv.set_index('date')
	residuals_csv.index = pd.to_datetime(residuals_csv.index)

	arima_csv = pd.read_csv('/home/ubuntu/csvs/arima.csv', parse_dates=True, infer_datetime_format=True)
	arima_csv = arima_csv.set_index('date')
	arima_csv.index = pd.to_datetime(arima_csv.index)

	ts_date_range = pd.date_range(start=date_ranges[12][0], end=date_ranges[15][-1])
	etf_df = pd.DataFrame(etfs[etf])
	true_price = etf_df[etf_df.index.isin(ts_date_range)]
	true_price = true_price.reset_index()
	true_price.columns = ['ds', 'y']

	arima_model = Prophet()
	arima_model.fit(true_price)

	future_date_range = [len(date_range) for date_range in date_ranges[16:]]

	future = arima_model.make_future_dataframe(periods = sum(future_date_range))
	forecast = arima_model.predict(future)

	prediction_date_range = pd.date_range(start=date_ranges[16][0], end=date_ranges[-1][-1])
	predicted_prices = np.array(forecast[forecast['ds'].isin(prediction_date_range)]['yhat'])
	predictor_year = pd.date_range(start = date_ranges[12][0], end = date_ranges[15][-1])

	scaler = StandardScaler()
	macro_frame = macro[macro.index.isin(predictor_year)]
	macro_unscaled = macro_frame.values
	macro_array = scaler.fit_transform(macro_unscaled)
	macro_predict = np.reshape(macro_array, (macro_array.shape[0], macro_array.shape[1], 1))
	residuals_array = gru_model.predict(macro_predict)
	predicted_residuals = (residuals_array.reshape(residuals_array.shape[1], residuals_array.shape[0])[0]).tolist()

	predicted_residuals.pop()

	final_predictions = (np.array(predicted_prices) + np.array(predicted_residuals)).tolist()

	predictions_csv[etf] = final_predictions
	predictions_csv.to_csv('/home/ubuntu/csvs/predictions.csv')

	arima_csv[etf] = predicted_prices.tolist()
	arima_csv.to_csv('/home/ubuntu/csvs/arima.csv')

	residuals_csv[etf] = predicted_residuals
	residuals_csv.to_csv('/home/ubuntu/csvs/residuals.csv')

	return_frame = pd.DataFrame(index=prediction_date_range)
	return_frame['price'] = predicted_prices
	return_frame['residual'] = predicted_residuals
	return_frame['prediction'] = final_predictions

	return return_frame

if __name__ == '__main__':
	main()
