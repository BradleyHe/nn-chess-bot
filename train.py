import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import chessboard
import pickle
import os

from keras import backend 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def plot_history(history):
  plt.style.use('ggplot')
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  mae = history.history['mean_absolute_error']
  val_mae = history.history['val_mean_absolute_error']
  x = range(1, len(loss) + 1)

  plt.figure(figsize=(12,5))
  plt.subplot(1,2,1)
  plt.plot(x, loss, 'b', label='Training MSE')
  plt.plot(x, val_loss, 'r', label='Validation MSE')
  plt.xticks(np.arange(0, 50, 5))
  plt.title('Training and validation MSE (loss)')
  plt.legend()
  
  plt.subplot(1,2,2)
  plt.plot(x, mae, 'b', label='Training loss MAE')
  plt.plot(x, val_mae, 'r', label='Validation loss MAE')
  plt.xticks(np.arange(0, 50, 5))
  plt.title('Training and validation MAE')
  plt.legend()
  plt.savefig('plot.png')

def map_to_evaluation(x, target_min=-255, target_max=255):
  x02 = backend.tanh(x) + 1 # x in range(0,2)
  scale = (target_max - target_min) /  2.
  return x02 * scale + target_min

def main():
  rows = 1000000

  print('\n__________________________\nLoading data...\n__________________________')

  # first time running
  # df = pd.read_csv('chessData.csv', nrows=rows)
  # x_all_data = np.empty((rows, 18, 8, 8), dtype=np.int8)
  # y_all_data = np.empty((rows), dtype=np.float32)

  # for fen,score,idx in list(zip(df['FEN'], df['Evaluation'], df.index)):
  #   if idx % 100000 == 0: print(idx)
  #   x_all_data[idx], y_all_data[idx] = chessboard.generate_one_hot_encoding(fen, str(score))

  # x_train, x_valid, y_train, y_valid = train_test_split(x_all_data, y_all_data, shuffle=None, test_size=0.2)

  # os.makedirs('data', exist_ok=True)
  # np.save('data/x_train.npy', x_train)
  # np.save('data/y_train.npy', y_train)
  # np.save('data/x_valid.npy', x_valid)
  # np.save('data/y_valid.npy', y_valid)

  # subsequent runs
  x_train = np.load('data/x_train.npy')
  y_train = np.load('data/y_train.npy')
  x_valid = np.load('data/x_valid.npy')
  y_valid = np.load('data/y_valid.npy')

  print('\n__________________________\nTraining model...\n__________________________')

  backend.set_image_dim_ordering('th')

  model = Sequential()
  model.add(Conv2D(50, (3, 3), activation='relu', input_shape=(18, 8, 8)))
  model.add(Conv2D(100, (3, 3), activation='relu'))
  model.add(Conv2D(200, (3, 3), activation='relu'))
  model.add(Conv2D(400, (2, 2), activation='relu'))
  model.add(Flatten())
  model.add(Dense(1, activation=map_to_evaluation))
  model.summary()

  os.makedirs('checkpoint', exist_ok=True)
  filepath = 'checkpoint/epoch_{epoch:02d}-mse_{val_loss:.2f}.hdf5'
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True)
  callbacks_list = [checkpoint]

  model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error', 'mean_absolute_error'])
  history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=50, callbacks=callbacks_list)

  plot_history(history)

if __name__ == '__main__':
  main()