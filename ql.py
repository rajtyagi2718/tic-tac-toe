from transposition import Set
from rl import RLSelfPlay

class QSelfPlay(RLSelfPlay):

    def evaluate_episode(self):
        """Gradient descent. Shift values toward BEST estimated afterstate.

        Undo board moves on each step. Reward is zero for non leaf nodes.
        Reward G is effectively value of action with best afterstate value.
        This is off policy i.e. not necessarily follow actual return value."""
        G = self.board.utility()
        self.visits[self.board] += 1
        self.values[self.board] = G
        for _ in range(self.board.moves()):
            self.board.pop()
            self.visits[self.board] += 1
            G = self.get_best_value()
            delta = self.alpha * (G - self.values[self.board])
            self.values[self.board] += delta

    def episode_delta(self):
        """Return max absolute change in values. Don't change values."""
        G = self.board.utility()
        max_delta = G - self.values[self.board]

        for _ in range(self.board.moves()):
            self.board.pop()
            G = self.get_best_value()
            delta = self.alpha * (G - self.values[self.board])
            max_delta = max(abs(delta), max_delta)

        return max_delta

class QSSelfPlay(QSelfPlay):

    def __init__(self, gamma=1, alpha=.5, epsilon=15, depth=3, values=None,
                 visits=None):
        super().__init__(gamma, alpha, epsilon, values, visits)
        self.search_tree = Set()
        self.depth = depth

    def evaluate_episode(self):
        """Gradient descent. Shift values toward BEST estimated afterstate.

        Find best with minimax search on values, cutoff at given depth.
        Depth 1 is equivalent to Q learning.

        Undo board moves on each step. Reward is zero for non leaf nodes.
        Reward G is effectively value of action with best afterstate value.
        This is off policy i.e. not necessarily follow actual return value."""
        G = self.board.utility()
        self.visits[self.board] += 1
        self.values[self.board] = G
        for _ in range(self.board.moves()):
            self.board.pop()
            self.visits[self.board] += 1
            G = self.get_best_value()
            delta = self.alpha * (G - self.values[self.board])
            self.values[self.board] += delta

    def episode_delta(self):
        """Return max absolute change in values. Don't change values."""
        G = self.board.utility()
        max_delta = G - self.values[self.board]

        for _ in range(self.board.moves()):
            self.board.pop()
            G = self.get_best_value()
            delta = self.alpha * (G - self.values[self.board])
            max_delta = max(abs(delta), max_delta)

        return max_delta

    def get_best_value(self):
        self.search_tree.clear()
        return self.explore(self.depth)

    def explore(self, depth):
        if self.cutoff_test(depth):
            return self.values[self.board]

        if self.board.turn() == 1:
            best = max
            value = -1
        else:
            best = min
            value = 1

        for action in self.board.get_actions():
            self.board.push(action)
            child = self.explore(depth-1)
            value = best(child, value)
            self.board.pop()

        return value

    def cutoff_test(self, depth):
        """Return True if depth is 0, board already explored, or terminal."""
        return (not depth or self.board in self.search_tree or
                self.board.is_terminal())
