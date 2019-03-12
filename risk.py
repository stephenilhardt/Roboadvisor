import numpy as np
import pandas as pd

import pandas_datareader.data as pdr
import fix_yahoo_finance as yf

from fake_useragent import UserAgent
from bs4 import BeautifulSoup

yf.pdr_override()

def parse_risk_data(ticker):
	ua = UserAgent()
	user_agent = {'User-agent': ua.random}

	response = requests.get('https://finance.yahoo.com/quote/{}/risk?p={}'.format(ticker,ticker), headers=user_agent)
    soup = BeautifulSoup(response.text, 'lxml')
    
    risk = soup.find_all('span', {'data-reactid': span_list})[4:]
    
    risk_list = []
    
    for data in risk:
        risk_list.append(data.text)

    return risk_list

def get_risk_data(tickers):
	risk_data = []

	for ticker in tickers:
		risk_list = parse_risk_data(ticker)
    	risk_data.append(risk_list)

    return risk_data

if __name__ == '__main__':
	get_risk_data()