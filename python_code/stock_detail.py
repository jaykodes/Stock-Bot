import numpy as np
import pandas as pd
import datetime as dt
from stock import stock
from itertools import combinations

class detailed_stock(stock):
	
	def __init__(self, stock_sym, stock_name, sma_periods=[10, 20, 50]):
		super().__init__(stock_sym, stock_name, sma_periods)

	def combination_details(self, data, combs):

		def difference(val_one, val_two):
			return val_one - val_two;

		para_cols = ["Date"]

		for i in list(combs):
			data[f"{i[0]} vs {i[1]}"] = list(map(difference, data[f"{i[0]}"], data[f"{i[1]}"]))
			para_cols.append(f"{i[0]} vs {i[1]}")

		return data, para_cols

	def get_detailed_chart_data(self, feature, time_period):

		chart_data = self.chart_data(feature, time_period)
		return chart_data

	def get_detailed_sma_data(self, feature, time_period):

		chart_data = self.chart_data(feature, time_period)
		sma_data = self.sma_data(feature, time_period, self.sma_periods)

		for sma_period in self.sma_periods:
			sma_data.rename(columns={sma_period : f"{sma_period}-SMA"}, inplace=True)

		sma_data = pd.concat([chart_data.set_index("Date"), sma_data.set_index("Date")], axis=1, join="inner").reset_index()
		sma_combs = combinations(sma_data.columns[1:], 2)
		sma_data, para_cols = self.combination_details(sma_data, sma_combs)
		
		return sma_data

	def get_detailed_data(self, feature, time_period):

		chart_data = self.get_detailed_chart_data(feature, time_period)
		sma_data = self.get_detailed_sma_data(feature, time_period)

		data = pd.concat([chart_data.set_index("Date"), sma_data.set_index("Date")], axis=1, join="inner").reset_index()
		return data

def main():
	aapl = detailed_stock("AAPL", "Apple")
	print(aapl.get_detailed_data("Adj Close", 90))

if __name__ == '__main__':
	main()