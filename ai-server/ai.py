import random


def pick_next_move(board, player):
    # TODO:  AI behavior here.
    # Return: instance of BaseMove (ForwardMove, CaptureMove, or ComboCaptureMove)

    #next_move = get_next_move(board, player, 'minmax_simple') # For regular minmax search
    next_move = get_next_move(board, player, 'abpruning')  # For minmax algo with alpha beta pruning
    return next_move


def heuristic_function(board , player):
    """
    Reward for right move choice. The bigger is heuristic after this test - the better.
    Counted on the difference in number of opponent's and player's figures after a move.
    :return: grade for a move
    """
    if player ==1:
        oponent = 2
    else:
        oponent = 1
    sq_player = board.get_player_squares(player)
    sq_oponent = board.get_player_squares(oponent)
    n_player = len(sq_player)
    n_oponent = len(sq_oponent)

    n_pl_kings = 0
    n_op_kings = 0
    for square in  sq_player:
        if board.piece_class[square] == 2:
            n_pl_kings += 1
    for sq in sq_oponent:
        if board.piece_class[sq] == 2:
            n_op_kings += 1

    return (n_player-n_oponent) + (n_pl_kings-n_op_kings)


def minimax(move, board, depth, player,  st_player, list_h):
    """
    Takes as origin node the state in move and buids a tree for all possible moves that come after this node.
    Then does minimax search to find heuristic function for each leaf inside the tree and gives the best result.
    :list_h: list of grades from heuristic function for a particular move
    :return: heuristic function result for a particular move, that  is given in function as a parameter
    """
    if depth == 0 or len(board.get_available_moves(player))==0:
        grade = heuristic_function(board, st_player)
        list_h.append(grade)
        return grade

    new_board = move.apply(board)
    moves = new_board.get_available_moves(player)

    if player == 1:
        best_value = float("-inf")
        for move in moves:
            v = minimax(move, new_board, depth-1, 2,  st_player, list_h)
            best_value = max(best_value, v)
        return best_value

    elif player ==2:
        best_value = float("-inf")
        for move in moves:
            v = minimax(move, new_board, depth-1, 1 , st_player, list_h)
            best_value = min(best_value, v)
        return best_value


def get_next_move(board, player, mode):
    """
    Returns best move after minimax search.
    :return: best move
    """
    list_h = []
    pos_moves = board.get_available_moves(player)
    future_states_grades = []
    pl = 1
    if player == 1:
        pl = 2

    for move in pos_moves:  # for each move create a tree with chosen depth and evaluate this move's value
        if mode == 'abpruning':
            # Parameters for alpha_beta: aplha_beta(origin, board, depth, -inf, inf, player, static_player, [])
            alpha = float("-inf")
            beta = float("inf")
            future_state = alpha_beta(move, board, 5, alpha, beta, pl, player, list_h)
        elif mode == 'minmax_simple':
            future_state = minimax(move, board, 4, pl, player, list_h)
        try:
            future_states_grades.append(max(list_h))
        except ValueError:
            return random.choice(pos_moves)

        list_h = []

    best_moves_indeces = [m for m in range(len(future_states_grades)) if future_states_grades[m] == max(future_states_grades)]
    # If maximum values of a few moves' grades are the same, pick a random move from them
    return pos_moves[random.choice(best_moves_indeces)]


def alpha_beta(move, board, depth, alfa, beta, player, st_player, answers):
    """
    Takes as origin node the state in move and buids a tree for all possible moves that come after this node.
    Then does minimax search with alfa-beta pruning to find heuristic function just for those leaves that are not prunned
    inside the tree and gives the best result.
    :return: heuristic function result for a particular move, that  is given in function as a parameter
    """
    if depth == 0 or len(board.get_available_moves(player))==0:
        grade = heuristic_function(board, st_player)
        answers.append(grade)
        return grade

    new_board = move.apply(board)
    moves = new_board.get_available_moves(player)
    if player == 1:
        # Maximizing player's winning points
        v = float("-inf")
        for move in moves:
            v = max(v, alpha_beta(move,new_board, depth-1, alfa, beta, 2, st_player, answers))
            alfa = max(alfa, v)
            if beta <= alfa:
                break # pruning
        return v

    elif player == 2:
        # Minimizing opponents's winning points
        v = float("-inf")
        for move in moves:
            v = min(v, alpha_beta(move,new_board, depth-1, alfa, beta, 1, st_player, answers))
            beta = min(beta, v)
            if beta<= alfa:
                break # pruning
        return v