from transposition import Table
from board import Board

class Tree:

    """Game state tree stored as dict map from board hashes to utility values.

    Root of complete game state tree is empty board. Each action during game
    step produces child board. Root has nine children.

    [X _ _] [_ X _] [_ _ X] [_ _ _] [_ _ _] [_ _ _] [_ _ _] [_ _ _] [_ _ _]
    [_ _ _] [_ _ _] [_ _ _] [X _ _] [_ X _] [_ _ X] [_ _ _] [_ _ _] [_ _ _]
    [_ _ _] [_ _ _] [_ _ _] [_ _ _] [_ _ _] [_ _ _] [X _ _] [_ X _] [_ _ X]

    Leaves are terminal boards.

    Tree stores boards by hash function on board array. Board array equates
    transpositions (boards with same state but different paths). Hash function
    equates symmetric boards. This leave three children of root.

        [X _ _] [_ X _] [_ _ _]
        [_ _ _] [_ _ _] [_ X _]
        [_ _ _] [_ _ _] [_ _ _]

    Tree maps boards to utility values. Given parent board, afterstate values
    are utility values of child boards.

    AI searches tree for policy insight. By analyzing afterstate values, best
    action can be chosen.
    """

    def __init__(self, table=None):
        if table is None:
            table = Table()
        self.table = table

    ## Search methods ##

    def get_action_values(self, board):
        """Return pairs of action, afterstate value."""
        result = []
        for action in board.get_actions():
            board.push(action)
            value = self.table[board]
            result.append((action, value))
            board.pop()
        return result

    def best_actions(self, board, action_values):
        """Return actions with highest (lowest) value for agent1 (agent2)."""
        best = max if board.turn() == 1 else min
        best_val = best(value for _,value in action_values)
        return [action for action,value in action_values if value == best_val]

    def norm_action_values(self, board, action_values):
        """Return values scaled within -1 to 1. Sort pairs by action number."""
        return sorted(action_values, key=lambda x: x[0])

    def get_best_actions(self, board):
        return self.best_actions(board, self.get_action_values(board))

    def get_norm_actions_values(self, board):
        return self.norm_action_values(board, self.get_action_values(board))
