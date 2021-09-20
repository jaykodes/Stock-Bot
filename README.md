# Stock-Bot

This stock bot uses a recurrent neural network to predict when to buy and short a stock. This is done by using the previous day data of each stock. The bot also uses technical indicators such as moving averages, moving average convergence/divergence (MACD), and slow stochastics to help with its prediction.

<h2> stock.py </h2>

Each company that in our portfolio is an stock object. Each stock object has the following attributes:

  1. Stock Symbol: This is the stock ticker that is used in the real world (AAPL, MSFT, FB, etc)
  2. Stock Name: This is the name of each company (Apple, Microsoft, Facebook, etc)
  3. Stock Data: This is a Pandas dataframe where each row represents a day of action in the real world. Each row typically contains the high, low, open, close, and adjusted close price of each day.
  4. SMA Periods: This is the Simple Moving Average time ranges that the user wants to use
  5. EMA Periods: This is the Exponential Moving Average time ranges that the user wants to use
  6. MACD Periods: This is the Moving Average Convergence/Divergence (MACD) time ranges that the user wants to use
  7. Sto Periods: This is the Slow Stochastics time ranges that the user wants to use

The stock.py file also contain various function that calculate the SMA, EMA, MACD, and Slow Stochastics values. It finally has a function to plot all of this information in a repsonsive and interactive dashboard.

View the files in stock_data for information grabbed.

<h2> stock_detail.py </h2>

The stock_detail.py file contains the information about another stock object called detailed_stock where its parent is a stock object. This file starts preparing the data for the machine learning model to read. This function will take the SMA, EMA, MACD, and Slow Stochastics values and give them 1s and 0s depending on if the values are greater than one another in their respective sections.

For example: If the 20-day SMA = 5 and the 30-day SMA = 10 the function created will output 0 because if the longer-term SMA is greater than the short-term SMA then it is an indication of a downtrend.

Finally, the object has a function that returns a dataframe containing all of the information and their 1s and 0s value.

View the files in detailed_stock_data for information grabbed.

<h2> stock_portfolio.py </h2>

The stock_portfolio.py introduced yet another class called stock_portfolio. The stock_portfolio object represents a users stock portfolio. It then takes the stock portfolio and gets the information of each stock using the detailed_stock and stock objects.
