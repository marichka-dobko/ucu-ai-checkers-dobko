#!/usr/bin/env python3

import argparse
import logging
import math
import json
import random
import sys
from collections import Counter
from datetime import datetime
from urllib.parse import urljoin

import numpy as np
import matplotlib.pyplot as plt
import requests
from libcheckers import BoardConfig
from libcheckers import serialization as ser
from libcheckers.enum import Player, PieceClass, GameOverReason
from libcheckers.movement import Board, ForwardMove, CaptureMove, ComboCaptureMove
from libcheckers.utils import index_to_coords


# How long we should wait for the server to come up with the next move.
SERVER_REQUEST_TIMEOUT_SEC = 10

# The minimum number of moves each player must make to consider the game a draw.
MAX_MOVES = 100

# GUI delay after visualizing each move.
MOVE_VISUALIZATION_DELAY_SEC = 0.25

# GUI delay before starting and ending each game.
GAME_OVER_VISUALIZATION_DELAY_SEC = 1.5


logger = None
run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')


def parse_command_line_args(args):
    """
    Parse the command-line arguments into a structured object.
    """

    parser = argparse.ArgumentParser(
        description='Checkers game arena runner',
        epilog='Example: arena.py compete --gui http://localhost:5001 http://localhost:5002',
    )
    subparsers = parser.add_subparsers(dest='command', help='Mode to run')

    # Parser for 'compete' command.
    compete_parser = subparsers.add_parser(
        'compete',
        help='Run the arena in competitive mode between two player servers',
    )
    compete_parser.add_argument(
        'white_server',
        help='URL of the AI server playing white (or "human" to play on keyboard)',
    )
    compete_parser.add_argument(
        'black_server',
        help='URL of the AI server playing black (or "human" to play on keyboard)',
    )
    compete_parser.add_argument(
        '--num-games',
        metavar='n',
        required=False,
        type=int,
        default=1,
        help='Number of games to run (default: 1)',
    )
    compete_parser.add_argument(
        '--gui',
        dest='gui',
        required=False,
        action='store_true',
        help='Visualize the board while running the game',
    )

    # Parser for 'replay' command.
    replay_parser = subparsers.add_parser(
        'replay',
        help='Replay a previously saved game',
    )
    replay_parser.add_argument('replay_file', help='Game replay file name (*.replay)')

    parsed_args = parser.parse_args(args)
    if parsed_args.command is None:
        parser.print_help()
        sys.exit(2)

    return parsed_args


def setup_logging():
    global logger
    log_format = '%(asctime)s | %(levelname)8s | %(message)s'
    handlers = [
        logging.FileHandler('arena-{0}.log'.format(run_timestamp), encoding='utf-8'),
        logging.StreamHandler(stream=sys.stdout)
    ]
    logging.basicConfig(handlers=handlers, level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)


def create_plot(window_title):
    fig, ax = plt.subplots(figsize=(5, 5), num=window_title)
    fig.tight_layout()
    return ax


def run_competition(args):
    """
    Run the arena in a 1v1 competition mode between the specified two AI servers
    for the specified number of rounds.
    """

    game_history = []
    plot = create_plot('Checkers Game Arena') if args.gui else None

    # Run the specified number of games.
    for game_num in range(args.num_games):
        msg = 'Starting game {0} of {1}, white: [{2}], black: [{3}]'
        msg = msg.format(game_num + 1, args.num_games, args.white_server, args.black_server)
        logger.info(msg)

        # Run a single game.
        moves, game_over_reason = run_game(args, plot)
        game_history.append((moves, game_over_reason))

        # Save game for future replay.
        game_filename = 'game-{0}-{1}.replay'.format(run_timestamp, game_num + 1)
        save_game(moves, game_over_reason, game_filename)
        logger.info('Game saved to {0}'.format(game_filename))

    # Print the overall summary from all games.
    print('{0} Summary {0}'.format('-' * 25))
    for game_num, (moves, game_over_reason) in enumerate(game_history):
        msg = 'Game {0}: {1} turns, outcome: {2}'
        outcome_name = get_reason_message(game_over_reason)
        print(msg.format(game_num + 1, math.ceil(len(moves) / 2), outcome_name))

    outcome_counter = Counter(game_over_reason for _, game_over_reason in game_history)
    print()
    print('White wins:', outcome_counter[GameOverReason.WHITE_WON])
    print('Black wins:', outcome_counter[GameOverReason.BLACK_WON])
    print('Draws     :', outcome_counter[GameOverReason.DRAW])


def run_game(args, plot):
    """
    Run a single game of the 1v1 competition.
    """

    def end_game(moves, reason, message=None):
        msg = message if message else 'Game over: {0}'.format(get_reason_message(reason))
        logger.info(msg)
        if plot:
            plt.pause(GAME_OVER_VISUALIZATION_DELAY_SEC)
        return moves, reason

    board = get_starting_board()
    render_board(board, plot)
    moves = []

    if plot:
        plt.pause(GAME_OVER_VISUALIZATION_DELAY_SEC)

    for move_number in range(1, MAX_MOVES + 1):
        # White move.
        game_over = board.check_game_over(Player.WHITE)
        if game_over:
            return end_game(moves, GameOverReason.BLACK_WON)

        white_move = get_player_move(move_number, board, Player.WHITE, args.white_server)
        moves.append(white_move)
        render_move(board, plot, white_move)
        board = white_move.apply(board)

        # Black move.
        game_over = board.check_game_over(Player.BLACK)
        if game_over:
            return end_game(moves, GameOverReason.WHITE_WON)

        black_move = get_player_move(move_number, board, Player.BLACK, args.black_server)
        moves.append(black_move)
        render_move(board, plot, black_move)
        board = black_move.apply(board)

    return end_game(moves, GameOverReason.DRAW, message='Game over: maximum limit of moves reached')


def replay_game(args):
    """
    Replay a previously saved game log file.
    """

    logger.info('Replaying game: {0}'.format(args.replay_file))

    moves, game_over_reason = load_game(args.replay_file)
    plot = create_plot('Game Replay: {0}'.format(args.replay_file))

    board = get_starting_board()
    render_board(board, plot)
    plt.pause(GAME_OVER_VISUALIZATION_DELAY_SEC)

    for move_num, move in enumerate(moves):
        player_name = 'white' if not move_num % 2 else 'black'
        turn_number = int(math.ceil((move_num + 1) / 2))
        logger.info('Move {0:3d}: {1} plays {2}'.format(turn_number, player_name, move))
        render_move(board, plot, move)
        board = move.apply(board)

    logger.info('Game over: {0}'.format(get_reason_message(game_over_reason)))
    plt.pause(GAME_OVER_VISUALIZATION_DELAY_SEC)


def load_game(game_filename):
    """
    Load the game moves and outcome from a replay file.
    """

    with open(game_filename) as gf:
        game_data = json.load(gf)
    moves = [ser.load_move(move) for move in game_data['moves']]
    game_over_reason = ser.load_game_over_reason(game_data['gameOverReason'])
    return moves, game_over_reason


def save_game(moves, game_over_reason, game_filename):
    """
    Save the game moves and outcome to a replay file.
    """

    game_data = {
        'moves': [ser.save_move(move) for move in moves],
        'gameOverReason': ser.save_game_over_reason(game_over_reason),
    }
    with open(game_filename, 'w') as gf:
        json.dump(game_data, gf, indent=4)


def get_starting_board():
    """
    Get an instance of the checkers game board representing the initial
    layout of the game pieces.
    """

    board = Board()
    for index in range(31, 51):
        board.add_piece(index, Player.WHITE, PieceClass.MAN)
    for index in range(1, 21):
        board.add_piece(index, Player.BLACK, PieceClass.MAN)
    return board


def get_player_move(move_num, board, player, server):
    """
    Retrieve the next move from the keyboard input or an AI server.
    Fall back to a random move if an error occurs.
    """

    allowed_moves = board.get_available_moves(player)

    if server == 'human':
        move = get_player_move_from_keyboard(board, player, allowed_moves)
    else:
        move = get_player_move_from_server(board, player, allowed_moves, server)

    if move not in allowed_moves:
        msg = ('Player {0} picked a move that is not allowed ({1}). '
               'Picking a random move instead')
        logger.warning(msg)
        move = random.choice(allowed_moves)

    logger.info('Move {0:3d}: {1} plays {2}'.format(move_num, get_player_name(player), move))
    return move


def get_player_move_from_keyboard(board, player, allowed_moves):
    """
    Receive the next move from the keyboard input.
    """

    def move_sort_key(move):
        if isinstance(move, ForwardMove):
            return move.start_index
        if isinstance(move, CaptureMove):
            return move.start_index
        if isinstance(move, ComboCaptureMove):
            return move.moves[0].start_index

    # If we have only one available move, no need to ask the user to decide.
    if len(allowed_moves) == 1:
        return allowed_moves[0]

    # List available moves, each having a unique number.
    print()
    print('Pick a move for {0}:'.format(get_player_name(player)))
    sorted_moves = sorted(allowed_moves, key=move_sort_key)
    for i, move in enumerate(sorted_moves):
        print('{0:2d}) {1}'.format(i + 1, move))

    # Ask for input until we get a valid selected move.
    picked_move = None
    while not picked_move:
        try:
            picked_number = input('Move number: ')
            picked_move = sorted_moves[int(picked_number) - 1]
        except Exception:
            print('Input error: must pick a number from the list')

    return picked_move


def get_player_move_from_server(board, player, allowed_moves, server_url):
    """
    Retrieve and parse the next move from an AI server.
    """

    player_name = ser.save_player(player)
    payload = {
        'board': ser.save_board(board),
        'playerTurn': player_name,
    }
    url = urljoin(server_url, 'move')

    try:
        response = requests.post(url, json=payload, timeout=SERVER_REQUEST_TIMEOUT_SEC)
        if response.status_code != 200:
            msg = ('Player {0}: server has responded with an unexpected status code: {1}. '
                   'Picking a random move instead')
            logger.warning(msg.format(player_name, response.status_code, response.content))
            return random.choice(allowed_moves)
    except Exception as ex:
        msg = ('Player {0}: Error when requesting next move from the server: {1}. '
               'Picking a random move instead')
        logger.warning(msg.format(player_name, ex))
        return random.choice(allowed_moves)

    try:
        return ser.load_move(response.json())
    except Exception:
        msg = ('Player {0}: Unable to parse the move returned by the server ({1}). '
               'Picking a random move instead')
        logger.warning(msg.format(player_name, response.content))
        return random.choice(allowed_moves)


def get_player_name(player):
    return ser.save_player(player)


def get_reason_message(game_over_reason):
    if game_over_reason == GameOverReason.WHITE_WON:
        return 'white won'
    elif game_over_reason == GameOverReason.BLACK_WON:
        return 'black won'
    elif game_over_reason == GameOverReason.DRAW:
        return 'draw'


def get_piece_symbol(piece_class):
    man_symbol = '\u26c2'
    king_symbol = '\u26c3'
    return king_symbol if piece_class == PieceClass.KING else man_symbol


def render_board(board, plot):
    """
    Draw the game board and pieces on a GUI window.
    """

    if not plot:
        # GUI is disabled.
        return

    board_matrix = np.zeros((BoardConfig.board_dim, BoardConfig.board_dim, 3))

    # "Black" square color.
    board_matrix += 0.6
    # "White" square color.
    board_matrix[::2, ::2] = 1.0
    board_matrix[1::2, 1::2] = 1.0

    # Board background.
    plot.cla()
    plot.imshow(board_matrix, interpolation='nearest')
    plot.set(xticks=[], yticks=[])

    white_color = '#F8F8F8'
    black_color = '#303030'
    index_color = '#787878'

    for index in range(1, BoardConfig.total_squares + 1):
        # Square indexes.
        row, column = index_to_coords(index)
        plot.text(column - 1.35, row - 0.65, str(index), color=index_color, size=7, ha='center', va='center')

        # Game pieces.
        if board.owner[index]:
            color = white_color if board.owner[index] == Player.WHITE else black_color
            symbol = get_piece_symbol(board.piece_class[index])
            plot.text(column - 1, row - 1, symbol, color=color, size=25, ha='center', va='center')

    plt.pause(MOVE_VISUALIZATION_DELAY_SEC)


def render_move(board, plot, move):
    """
    Pseudo-animate a single move as it is being applied to the board.
    """

    if not plot:
        # GUI is disabled.
        return

    white_move_color = '#FFF080'
    black_move_color = '#1878BA'

    if isinstance(move, ComboCaptureMove):
        for single_move in move.moves:
            render_move(board, plot, single_move)
            board = single_move.apply(board)
        return

    symbol = get_piece_symbol(board.piece_class[move.start_index])
    color = white_move_color if board.owner[move.start_index] == Player.WHITE else black_move_color

    # Move start.
    row, column = index_to_coords(move.start_index)
    plot.text(column - 1, row - 1, symbol, color=color, alpha=1.0, size=25, ha='center', va='center')

    # Move end.
    row, column = index_to_coords(move.end_index)
    plot.text(column - 1, row - 1, symbol, color=color, alpha=0.5, size=25, ha='center', va='center')

    plt.pause(MOVE_VISUALIZATION_DELAY_SEC)
    new_board = move.apply(board)
    render_board(new_board, plot)
    plt.pause(MOVE_VISUALIZATION_DELAY_SEC)


def main():
    """
    Arena entry point.
    """

    args = parse_command_line_args(sys.argv[1:])
    setup_logging()

    if args.command == 'compete':
        run_competition(args)
    if args.command == 'replay':
        replay_game(args)


if __name__ == '__main__':
    main()
