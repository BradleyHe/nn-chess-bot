import numpy as np

import chess

from math import floor

def generate_all_next_positions(fen):
	b = chess.Board(fen)
	positions = []
	
	for move in b.legal_moves:
		b.push(move)
		positions.append((move, b.fen()))
		b.pop()

	return positions

def generate_one_hot_encoding(fen, score=None):
	b = chess.Board(fen)
	pos = np.zeros([18, 8, 8])

	# ------------------------------------------------
	# ----------     position processing 		----------
	# ------------------------------------------------
	idx_to_piece = {0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING}

	# flip board if black's turn
	if b.turn == chess.WHITE:
		color_dict = {1: chess.WHITE, -1: chess.BLACK}
	else:
		color_dict = {-1: chess.WHITE, 1: chess.BLACK}

	for piece, color in [(p,c) for p in idx_to_piece.keys() for c in color_dict.keys()]:
		for square in b.pieces(idx_to_piece[piece], color_dict[color]):
			pos[piece][7-floor(square/8)][square%8] = color

	# -------------------------------------------------
	# ---------- attacked squares processing ----------
	# -------------------------------------------------
	for sq_num in range(64):
		if b.piece_type_at(sq_num) == type(None):
			continue
		color = b.color_at(sq_num)
		for square in b.attacks(sq_num):
			if b.turn == chess.WHITE:
				pos[6+int(not color)*6+b.piece_type_at(sq_num)-1][7-floor(square/8)][square%8] = 1
			elif b.turn == chess.BLACK:
				pos[6+int(color)*6+b.piece_type_at(sq_num)-1][7-floor(square/8)][square%8] = 1

	# flip board if black's turn
	if b.turn == chess.BLACK:
		for x in range(len(pos)):
			pos[x] = np.flip(pos[x])

	if score == None:
		return pos

	# -------------------------------------------------
	# ---------- 	  	score processing 			 ----------
	# -------------------------------------------------
	adj_score = 0

	# forced checkmate
	if score == '#+1' or score == '#-1':
		adj_score = int(score[1] + '255')
	elif score[0] == '#':
		adj_score = int(score[1] + '127')
	else:
		score = ''.join([c for c in score if c in '-+0123456789'])
		adj_score = float(score)/100
		if adj_score < -63:
			adj_score = -63
		elif adj_score > 63:
			adj_score = 63

	if b.turn == chess.BLACK:
		adj_score *= -1

	return pos,adj_score

def main():
	print(generate_one_hot_encoding('8/4K3/4Q3/8/8/8/8/4k3 w - - 0 1'))

if __name__ == '__main__':
	main()
