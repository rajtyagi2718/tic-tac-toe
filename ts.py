import numpy as np

from transposition import Set
from rl import RLSelfPlay

class TSSelfPlay(RLSelfPlay):

    def __init__(self, gamma=1, alpha=.5, epsilon=15, depth=3, values=None,
                 visits=None):
        super().__init__(gamma, alpha, epsilon, values, visits)
        self.depth = depth
        self.search_tree = Set()
        self.max_delta = 0

    def run_episode(self, greedy=False, evaluate=True, delta=False):
        self.board.reset()
        while not self.board.is_terminal():
            self.search_tree.clear()
            action = self.policy(greedy, evaluate, delta)
            self.board.push(action)

    def policy(self, greedy, evaluate, delta):
        if not greedy and not self.epsilon_greedy():
            action = self.random_policy()
        else:
            self.explore(self.depth, evaluate, delta)
            action = self.greedy_policy()
        return action

    def get_episode_delta(self):
        self.max_delta = 0
        self.run_episode(greedy=True, evaluate=False, delta=True)
        return self.max_delta

    def evaluate_tree_state(self, G):
        self.visits[self.board] += 1
        self.values[self.board] += self.alpha * (G - self.values[self.board])

    def tree_state_delta(self, G):
        delta = (G - self.values[self.board]) * self.alpha
        self.max_delta = max(abs(delta), self.max_delta)

    def explore(self, depth, evaluate=True, delta=False):
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

        if evaluate:
            self.evaluate_tree_state(value)
        if delta:
            self.tree_state_delta(value)

        return value

    def cutoff_test(self, depth):
        if not depth or self.board in self.search_tree:
            return True
        if self.board.is_terminal():
            self.values[self.board] = self.board.utility()
            return True
        return False
