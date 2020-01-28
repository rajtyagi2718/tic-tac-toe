from board import Board
from tree import Tree

class DPTree(Tree):
    """
    Recursively complete game tree. Symmetries collide. Backup leaf values.

    Explore recurses over board, push action for child, pop for parent.
    """

    def __init__(self, table=None, board=None):
        super().__init__(table)
        if table is None:
            if board is None:
                board = Board()
            self.board = board
            self.explore()
            del self.board

    def explore(self):
        """Recursively add child boards to table, backup values to parent."""

    def cutoff_test(self):
        """Return True if board is transposition or terminal, else False.

        If board is tranposition, it was explored from other parent and is
        in table. If board is terminal, it has no children, add to table.
        """
        if self.board in self.table:
            return True
        if self.board.is_terminal():
            self.table[self.board] = self.get_utility()
            return True
        return False

    def get_utility(self):
        """Return 0, 1, -1 for draw, agent1 win, agent2 win."""
        return self.board.utility()

class UniformTree(DPTree):
    """
    Opponent policy assumed to be random. Node utility is expected value.

    Agent1 chooses max value, agent2 min value. Leaf node value computed as 0,
    1, -1 for draw, agent1 win, agent2 win.

    No game theory or adversarial search used.
    """

    def explore(self):
        """Add children recursively depth first, backup mean value to parent.

        Bookkeep num children and mean value online. Check cutoff test to
        end recursion."""
        if self.cutoff_test():
            return self.table[self.board]

        num = mean = 0.0

        for action in self.board.get_actions():
            self.board.push(action)
            value = self.explore()
            num += 1
            mean += (value - mean) / num
            self.board.pop()

        self.table[self.board] = mean
        return mean

class DiscountTree(UniformTree):
    """
    Node values decay by rate gamma (10 percent) per ply from leaf.

    Leaf immediacy taken into account. Encourages longer games, higher variance
    against falliable opponents, same result as uniform against random player.
    """

    def __init__(self, table=None, board=None, gamma=.9):
        self.gamma = gamma
        super().__init__(board)

    def explore(self):
        """Add children depth first, backup discounted mean to parent. Recurse.

        Bookkeep num children and mean value online. Cutoff test to end
        recursion. Return discounted value."""
        if self.cutoff_test():
            return self.discount(self.table[self.board])

        num = mean = 0.0

        for action in self.board.get_actions():
            self.board.push(action)
            value = self.explore()
            num += 1
            mean += (value - mean) / num
            self.board.pop()

        self.table[self.board] = mean
        return self.discount(mean)

    def discount(self, value):
        """Discount value by rate gamma."""
        return self.gamma * value
