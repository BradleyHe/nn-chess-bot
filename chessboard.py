import chess
from math import floor
import numpy as np

def generate_all_moves(fen):
	return None

def generate_one_hot_encoding(fen, score):
	# ----- position processing -----

	b = chess.Board(fen)
	pos = np.zeros([6, 8, 8])
	piece_dict = {0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING}

	# flip board if black's turn
	if b.turn == chess.WHITE:
		color_dict = {-1: chess.WHITE, 1: chess.BLACK}
	else:
		color_dict = {1: chess.WHITE, -1: chess.BLACK}

	for piece, color in [(p,c) for p in piece_dict.keys() for c in color_dict.keys()]:
		for square in b.pieces(piece_dict[piece], color_dict[color]):
			pos[piece][7-floor(square/8)][square%8] = color
	
	# flip board if black's turn
	if b.turn == chess.BLACK:
		for x in range(len(pos)):
			pos[x] = np.flip(pos[x])


	# ----- score processing -----

	adj_score = 0

	# forced checkmate
	if score == '#+1' or score == '#-1':
		adj_score = int(score[1] + '255')
	elif score[0] == '#':
		adj_score = int(score[1] + '127')
	else:
		adj_score = int(score)
		if adj_score < -63:
			adj_score = -63
		elif adj_score > 63:
			adj_score = 63

	if b.turn == chess.BLACK:
		adj_score *= -1

	return pos,adj_score

def main():
	print(generate_one_hot_encoding('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'))

if __name__ == '__main__':
	main()
