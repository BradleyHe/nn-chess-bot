from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import chess
import chessboard
import sys
import logging
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# make the program shut up when we try and use it
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model
sys.stderr = stderr

from train import map_to_evaluation

def best_move_depth_3(model, fen):
	positions = chessboard.generate_all_next_positions(fen)
	evaluations = []

	for move, pos in positions:
		move_str = move.uci()
		evaluations.append((move_str, best_move_depth_2(model, pos)[1]))

	evaluations.sort(key=lambda x: x[1])
	return evaluations[0]

def best_move_depth_2(model, fen):
	positions = chessboard.generate_all_next_positions(fen)
	evaluations = []

	for move, pos in positions:
		move_str = move.uci()
		evaluations.append((move_str, best_move(model, pos)[1]))

	evaluations.sort(key=lambda x: x[1], reverse=True)
	return evaluations[0]

def best_move(model, fen):
	positions = chessboard.generate_all_next_positions(fen)

	if chess.Board(fen).is_checkmate():
		return (None, 255)
	elif len(positions) == 0:
		return (None, 0)

	evaluations = []

	for move, pos in positions:
		move_str = move.uci()
		evaluations.append((move_str, get_evaluation(model, pos)[0][0]))

	evaluations.sort(key=lambda x: x[1]) 
	return evaluations[0]

def get_evaluation(model, fen):
	x_data = np.empty((1, 18, 8, 8), dtype=np.int8)
	x_data[0] = chessboard.generate_one_hot_encoding(fen)
	return model.predict(x_data)

def main():
	model = load_model('/home/bradley/Desktop/nn-chess-bot/checkpoint/epoch_06-mse_174.24.hdf5', custom_objects={"map_to_evaluation": map_to_evaluation})
	move = best_move_depth_2(model, ' '.join(sys.argv[1:]))

	f = open('move.txt', 'w')
	f.write(move[0])
	f.close()

if __name__ == '__main__':
	main()