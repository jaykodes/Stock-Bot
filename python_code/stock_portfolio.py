import numpy as np
import pandas as pd
import pandas_datareader as web
import pathlib
import datetime as dt
from stock_detail import detailed_stock

class stock_portfolio():

	def __init__(self, feature="Close", start_year=2000, start_month=1, start_day=1, end_year=2020, end_month=12, end_day=31):
		
		self.stocks = {
			"MSFT": "Microsoft",
			"AAPL": "Apple",
			"AMZN": "Amazon",
			"GOOGL": "Google",
			"FB": "Facebook",
			"NFLX": "Netflix",
			"PYPL": "PayPal",
			"V": "Visa",
			"JPM": "JPMorgan",
			"NKE": "Nike",
			"IBM": "IBM",
			"GS": "Goldman Sachs",
			"MA": "Mastercard",
			"BAC": "Bank of America",
			"AXP": "American Express",
			"MS": "Morgan Stanley",
			"TSLA": "Tesla",
			"INTC": "Intel",
			"ADBE": "Adobe",
			"NVDA": "Nvidia"
		}

		self.feature = feature
		self.start_date = dt.datetime(start_year, start_month, start_day)
		self.end_date = dt.datetime(end_year, end_month, end_day)

	def get_stocks(self):
		return self.stocks

	def get_feature(self):
		return self.feature

	def get_start_date(self):
		return self.start_date

	def get_end_date(self):
		return self.end_date

	def produce_data(self):

		filename = "../stock_data"
		pathlib.Path(filename).mkdir(parents=True, exist_ok=True)

		for sym, name in self.stocks.items():

			print(sym, name)
			
			stock_data = web.get_data_yahoo(sym, start=self.start_date, end=self.end_date).reset_index()
			stock_data.to_excel(f"{filename}/{sym}_stock_data.xlsx", index=False)

	def produce_detailed_data(self):

		for sym, name in self.stocks.items():

			print(sym, name)
			stock = detailed_stock(sym, name)
			stock_data = stock.get_detailed_data(self.feature, stock.get_data_size())
			stock_data = stock_data.loc[:,~stock_data.columns.duplicated()]

			filename = "../stock_detailed_data"
			pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
			stock_data.to_excel(f"{filename}/{sym}_stock_detailed_data.xlsx", index=False)

	def view_stock(self, sym, time_period):

		stock = detailed_stock(sym, self.stocks[sym])
		stock.view_charts(self.feature, time_period)

def main():

	portfolio = stock_portfolio()
	portfolio.produce_data()
	portfolio.produce_detailed_data()

def not_main():

	portfolio = stock_portfolio()
	portfolio.view_stock("MSFT", 180)

if __name__ == '__main__':
	#main()
	not_main()
	pass