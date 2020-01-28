import numpy as np

class Search:
    """Generic class for searches. Agent queries policy for action."""

    def __init__(self, *args):
        """Analyze game board. Get utility of each action. Choose best."""

    def policy(self, game):
        """Return legal action for agent to take during current game step."""
        return np.random.choice(self.get_best_actions(game.board))

    def get_best_actions(self, board):
        """Return most valued actions."""

    def get_action_values(self, board):
        """Return action value pairs relative to current player."""

    def get_norm_action_values(self, board):
        """Return action value pairs, values normalized from -100 to 100."""

class RandomSearch(Search):
    """Random action chosen from game actions. Zero board analysis."""

    def get_best_actions(self, board):
        """Return all actions."""
        return board.get_actions()

    def get_action_values(self, board):
        """Return all actions valued at zero."""
        return [(action, 0) for action in self.get_best_actions()]

    def get_norm_action_values(self, board):
        """Values already normalized. Return get_action_values."""
        return self.get_action_values()

class TreeSearch(Search):
    """Search complete game tree map."""

    def __init__(self, tree):
        self.tree = tree
        super().__init__()

    def get_best_actions(self, board):
        """Query tree."""
        return self.tree.get_best_actions(board)

    def get_action_values(self, board):
        """Query tree."""
        return self.tree.get_action_values(board)

    def get_norm_action_values(self, board):
        """Query tree."""
        return self.tree.get_norm_action_values(board)
