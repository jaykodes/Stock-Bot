import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import style
import matplotlib.pyplot as plt
import datetime as dt
import mplfinance as mpf
from matplotlib.widgets import MultiCursor
import matplotlib.dates as mdates
from matplotlib.animation import ArtistAnimation
import pandas_datareader as web

style.use("ggplot")

class stock:

    def __init__(self, stock_sym, stock_name, sma_periods=[5, 10]):
        
        stock_data = pd.DataFrame()
        try:
            stock_data = pd.read_excel(f"../stock_data/{stock_sym}_stock_data.xlsx")
        except:
            stock_data = web.get_data_yahoo(stock_sym, start=dt.datetime(2000, 1, 1), end=dt.datetime(2021, 2, 20)).reset_index()
        
        self.stock_sym = stock_sym
        self.stock_name = stock_name
        self.stock_data = stock_data

        self.sma_periods = sma_periods

    def get_stock_sym(self):
        return self.stock_sym

    def get_stock_name(self):
        return self.stock_name

    def get_stock_data(self):
        return self.stock_data

    def get_sma_periods(self):
        return self.sma_periods

    def get_data_size(self):
        return len(self.stock_data.index) - 1 - max(self.sma_periods)

    def chart_data(self, feature, time_period):

        chart_data = self.stock_data[["Date", feature]].tail(time_period)
        return chart_data

    def get_sma_values(self, sma_stock_data, feature, time_period, sma_period):

        sma_values = []
            
        for i in range(len(sma_stock_data)-time_period, len(sma_stock_data)):
            sma_values.append(sma_stock_data.iloc[i-sma_period+1:i+1][feature].sum() / sma_period)

        return sma_values

    def sma_data(self, feature, time_period, sma_periods):

        sma_df = pd.DataFrame()
        sma_df["Date"] = self.stock_data["Date"].tail(time_period)

        for sma_period in sma_periods:
            
            sma_stock_data = self.stock_data.tail(sma_period+time_period).reset_index()
            sma_values = self.get_sma_values(sma_stock_data, feature, time_period, sma_period)
            sma_df[sma_period] = sma_values

        return sma_df

    def view_charts(self, feature, time_period):

        chart_data = self.chart_data(feature, time_period)
        sma_data = self.sma_data(feature, time_period, self.sma_periods)

        fig, axes = plt.subplots(nrows=3, figsize=(12, 9), gridspec_kw={'height_ratios': [6, 2, 2]}, sharex=True)
        
        axes[0].plot(chart_data["Date"], chart_data[feature], label=f"{self.stock_name} {feature}", marker='.')

        for sma_period in self.sma_periods:
            axes[0].plot(sma_data["Date"], sma_data[sma_period], label=f"{sma_period}-Day SMA")
        
        legend_one = axes[0].legend(loc=0, frameon=False)

        cursor = MultiCursor(fig.canvas, (axes), horizOn=False, vertOn=True, color="black", linewidth=1.0)

        def update_legend(event):

            mouse_x_val = event.xdata

            if (mouse_x_val is not None):
                mouse_x_date = mdates.num2date(mouse_x_val).replace(tzinfo=None)
                mouse_x_date = pd.Timestamp(mouse_x_date)

                data_str = ""
                
                closest_date = min(chart_data["Date"], key=lambda date: abs(date - mouse_x_date))
                data_str += f"Date: {closest_date}, "
                
                price = format(round(chart_data[chart_data["Date"] == closest_date][feature].values[0], 2), ".2f")
                data_str += f"Price: {price}, "

                for i, sma_period in enumerate(self.sma_periods):
                    sma_val = format(round(sma_data[sma_data["Date"] == closest_date][sma_period].values[0], 2), ".2f")
                    data_str += f"{sma_period}-SMA: {sma_val}, "

                print(data_str + "\n")

        fig.tight_layout()
        fig.canvas.mpl_connect('motion_notify_event', update_legend)
        fig.canvas.draw()

        plt.show()

def main():
    aapl = stock("GME", "Apple")
    aapl.view_charts("Close", 365)

if __name__ == '__main__':
    main()