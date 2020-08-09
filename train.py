import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import chessboard
import os

from keras import backend 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def plot_history(history):
  plt.style.use('ggplot')
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  x = range(1, len(loss) + 1)

  plt.plot(x, loss, 'b', label='Training loss')
  plt.plot(x, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.savefig('plot.png')

def map_to_evaluation(x, target_min=-255, target_max=255):
  x02 = backend.tanh(x) + 1 # x in range(0,2)
  scale = (target_max - target_min) /  2.
  return x02 * scale + target_min

rows = 100000

print('\n__________________________\nLoading data...\n__________________________')

df = pd.read_csv('chessData.csv', nrows=rows)
x_all_data = np.empty((rows, 6, 8, 8))
y_all_data = np.empty((rows))

for fen,score,idx in list(zip(df['FEN'], df['Evaluation'], df.index)):
	x_all_data[idx], y_all_data[idx] = chessboard.generate_one_hot_encoding(fen, score)

x_train, x_valid, y_train, y_valid = train_test_split(x_all_data, y_all_data, shuffle=None, test_size=0.2)

print('\n__________________________\nTraining model...\n__________________________')

model = Sequential()
model.add(Conv2D(384, kernel_size=3, activation='relu', input_shape=(6, 8, 8)))
model.add(Conv2D(192, kernel_size=2, activation='relu'))
model.add(Conv2D(96, kernel_size=2, activation='relu'))
model.add(Conv2D(48, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation=map_to_evaluation))

os.makedirs('checkpoint', exist_ok=True)
filepath = 'checkpoint/{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
callbacks_list = [checkpoint]

model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=100, callbacks=callbacks_list)

plot_history(history)