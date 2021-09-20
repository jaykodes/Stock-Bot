import numpy as np
import pandas as pd
from stock_portfolio import stock_portfolio
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pathlib
import pickle
import matplotlib.pyplot as plt

class stock_predictor():

	def __init__(self, seq_len=3, future_predict_period=1):

		self.seq_len = seq_len
		self.future_predict_period = future_predict_period
		self.portfolio = stock_portfolio()

	def get_seq_len(self):
		return self.seq_len

	def get_future_predict_period(self):
		return self.future_predict_period

	def get_epochs(self):
		return self.epochs

	def get_batch_size(self):
		return self.batch_size

	def get_portfolio(self):
		return self.portfolio

	def produce_data(self):

		self.portfolio.produce_data()
		print("--------------------")
		self.portfolio.produce_detailed_data()
		print("--------------------")

	def make_targets(self, stock_data, feature):

		def max_classifier(val_one, val_two):
			if (val_one > val_two):
				return 1
			return 0

		stock_data[f"Future {feature}"] = stock_data[feature].shift(-self.future_predict_period)
		stock_data["Target"] = list(map(max_classifier, stock_data[f"Future {feature}"], stock_data[feature]))
		return stock_data

	def preprocess_data(self, data, feature):

		data = data.drop(f"Future {feature}", axis=1)
		data = data.drop("Date", axis=1)

		for col in data.columns:
			if col != "Target":
				data[col] = data[col].pct_change()
				data.dropna(inplace=True)
				max_val = data[col].max()
				min_val = data[col].min()
				data[col] = (data[col] - min_val) / (max_val - min_val)

		data.dropna(inplace=True)

		sequential_data = []
		prev_days = deque(maxlen=self.seq_len)

		for i in data.values:
			prev_days.append([j for j in i[:-1]])
			if (len(prev_days) == self.seq_len):
				sequential_data.append([np.array(prev_days), i[-1]])

		random.shuffle(sequential_data)
		
		buys = []
		sells = []

		for seq, target in sequential_data:
			if (target == 0):
				sells.append([seq, target])
			elif (target == 1):
				buys.append([seq, target])

		random.shuffle(buys)
		random.shuffle(sells)

		lower = min(len(buys), len(sells))
		buys = buys[:lower]
		sells = sells[:lower]

		sequential_data = buys + sells
		random.shuffle(sequential_data)

		x = []
		y = []

		for seq, target in sequential_data:
			x.append(seq)
			y.append(target)

		return x, y

	def prepare_data(self):

		feature = self.portfolio.get_feature()
		stocks =  self.portfolio.get_stocks()

		train_x, train_y = [], []
		validation_x, validation_y = [], []

		for sym, name in stocks.items():

			stock_data = pd.read_excel(f"../stock_detailed_data/{sym}_stock_detailed_data.xlsx")
			stock_data = self.make_targets(stock_data, feature)
			
			times = stock_data.index.values
			last_5pct = times[-int(0.05*len(times))]

			train_data = stock_data[(stock_data.index < last_5pct)]
			validation_data = stock_data[(stock_data.index >= last_5pct)]

			temp_train_x, temp_train_y = self.preprocess_data(train_data, feature)
			temp_validation_x, temp_validation_y = self.preprocess_data(validation_data, feature)

			train_x = train_x + temp_train_x
			train_y = train_y + temp_train_y
			validation_x = validation_x + temp_validation_x
			validation_y = validation_y + temp_validation_y

		filename = "../stock_parameter_data"
		pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
		
		pickle_out = open(f"{filename}/train_x.pickle", "wb")
		pickle.dump(train_x, pickle_out)
		pickle_out.close()

		pickle_out = open(f"{filename}/train_y.pickle", "wb")
		pickle.dump(train_y, pickle_out)
		pickle_out.close()

		pickle_out = open(f"{filename}/validation_x.pickle", "wb")
		pickle.dump(validation_x, pickle_out)
		pickle_out.close()

		pickle_out = open(f"{filename}/validation_y.pickle", "wb")
		pickle.dump(validation_y, pickle_out)
		pickle_out.close()

	def make_model(self):

		name = f"{self.seq_len}-Seq-{self.future_predict_period}-Pred-{int(time.time())}"

		train_x = np.array(pickle.load(open("../stock_parameter_data/train_x.pickle", "rb")))
		train_y = np.array(pickle.load(open("../stock_parameter_data/train_y.pickle", "rb")))
		validation_x = np.array(pickle.load(open("../stock_parameter_data/validation_x.pickle", "rb")))
		validation_y = np.array(pickle.load(open("../stock_parameter_data/validation_y.pickle", "rb")))

		model = Sequential()

		model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation="tanh", return_sequences=False))
		model.add(Dense(2, activation="softmax"))

		opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

		model.compile(
			optimizer=opt,
			loss="sparse_categorical_crossentropy",
			metrics=["accuracy"]
		)

		tensorboard = TensorBoard(log_dir=f"logs/{name}")

		pathlib.Path("../models").mkdir(parents=True, exist_ok=True)
		filepath = "Stock_RNN_Final_{epoch:02d}-{val_accuracy:.3f}"
		checkpoint = ModelCheckpoint("../models/{}.hd5".format(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"))

		history = model.fit(
			train_x,
			train_y,
			batch_size=128,
			epochs=150,
			validation_data=(validation_x, validation_y),
			callbacks=[tensorboard]
		)

		plt.plot(history.history["loss"])
		plt.show()

def main():
	stock_model = stock_predictor()
	stock_model.produce_data()
	stock_model.prepare_data()
	stock_model.make_model()

if __name__ == '__main__':
	main()
	pass