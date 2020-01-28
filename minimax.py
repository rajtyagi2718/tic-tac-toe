from dp import DPTree

class MinimaxTree(DPTree):
    """
    Board values of optimal policy (Nash equilibrium). Opponent also optimal.

    Value equals max (min) of children if turn is agent1 (agent2).

    Neglects suboptimal play. For example, opening moves considered equal
    since optimal agent can always force draw. But middle position offers
    more possible winning paths against uniform or random player.
    """
    def explore(self):
        """Add children depth first, backup minimax value to parent. Recurse.

        Bookkeep min (max) value of children online. Check cutoff test to end
        recursion."""
        if self.cutoff_test():
            return self.table[self.board]

        if self.board.turn() == 1:
            value = -2
            best = max
        else:
            value = 2
            best = min

        for action in self.board.get_actions():
            self.board.push(action)
            child = self.explore()
            value = best(child, value)
            self.board.pop()

        self.table[self.board] = value
        return value

    def best_actions(self, board, action_values):
        """Return actions with afterstate values equal to parent value."""
        parent = self.table[board]
        return [action for action,child in action_values if child == parent]

class NegaminTree(MinimaxTree):
    """Same theory as minimax, but set values relative to turn of other agent.

    Exploit alternating turns. Evaluation always takes min of negated values
    of children."""

    def explore(self):
        """Add children depth first, backup minimax value to parent. Recurse.

        Bookkeep min negated child value online. Check cutoff test to end
        recursion."""
        if self.cutoff_test():
            return self.table[self.board]

        value = 1

        for action in self.board.get_actions():
            self.board.push(action)
            child = self.explore()
            value = min(-child, value)
            self.board.pop()

        self.table[self.board] = value
        return value

    def get_utility(self):
        """Return 0 for draw, 1 for either agent win. Pov of parent."""
        return abs(self.board.utility())

    def best_actions(self, board, action_values):
        """Return actions with afterstate values of negated parent value."""
        parent = -self.table[board]
        return [action for action,child in action_values if child == parent]

    def norm_action_values(self, board, action_values):
        """Return values scaled -1 to 1. Sort pairs by action number.

        Values already scaled. Want point of view of agent of parent board
        (given). So negate."""
        return sorted(((a,-v) for a,v in action_values), key=lambda x: x[0])
