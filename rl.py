import numpy as np
import os

from board import Board
from tree import Tree
from transposition import Table, DefaultTable

DATA_PATH = os.getcwd() + '/data/'

def get_random_value():
    return np.random.random() * np.random.choice((-1,1))

class RLSelfPlay:

    def __init__(self, gamma=1, alpha=.5, epsilon=1, values=None, visits=None):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        if values is None:
            values = DefaultTable()
        self.values = values
        if visits is None:
            visits = DefaultTable()
        self.visits = visits
        self.board = Board()

    ## Run methods ##

    def run(self, episodes):
        """Generate, evaluate given number of episodes."""
        for _ in range(episodes):
            self.run_episode()

    def run_episode(self):
        self.generate_episode()
        self.evaluate_episode()

    def get_episode_delta(self):
        """Generate, eval one episode. Return max change, don't alter value."""
        self.generate_episode(greedy=True)
        return self.episode_delta()

    ## Episode methods ##

    def generate_episode(self, greedy=False):
        """Take steps until board is terminal. Follow policy."""
        assert not self.board.moves(), (self.board)# assert board is reset
        while not self.board.is_terminal():
            action = self.policy(greedy)
            self.board.push(action)

    def evaluate_episode(self):
        """Backup state rewards according to particular rl algorithm."""

    def evaluate_episode_delta(self):
        """Backup state rewards, return max absolute change in values."""

    def episode_delta(self):
        """Return max absolute change in values. Don't change values."""

    ## Policy methods ##

    def policy(self, greedy=False):
        if not greedy and not self.epsilon_greedy():
            action = self.random_policy()
        else:
            action = self.greedy_policy()
        return action

    def random_policy(self):
        """Return random choice of actions."""
        return np.random.choice(self.board.get_actions())

    def greedy_policy(self):
        """Return random choice of best actions."""
        return np.random.choice(self.get_best_actions())

    def epsilon_greedy(self):
        """Return True with prob epsilon -- function of attr, visits.

        Define epsilon as a function of self.epsilon and number visits to
        current state. Satisfies GLIE with linear decay of exploration (random
        action) but greedy as visits approach infinity.
        """
        e = self.epsilon / (self.epsilon + self.visits[self.board])
        return np.random.random() >= e

    ## Search methods ##

    def get_best_items(self):
        """Return tuple: list actions with best afterstate, value itself."""
        items = []
        best = max if self.board.turn() == 1 else min

        for action in self.board.get_actions():
            self.board.push(action)
            items.append((action, self.values[self.board]))
            self.board.pop()

        best_value = best(items, key=lambda x: x[1])[1]
        best_actions = [act for act,val in items if val == best_value]

        return best_actions, best_value

    def get_best_actions(self):
        return self.get_best_items()[0]

    def get_best_value(self):
        return self.get_best_items()[1]

    ## Convergence runs ##

    def run_greedy_convergence(self, threshold=.01, interval=100, checks=20):
        total = 0
        delta = threshold
        while delta >= threshold:
            self.run_episodes(num_episodes=interval-checks, greedy=True)
            total += interval-checks
            for _ in range(checks):
                delta = self.run_episode_delta(greedy=True)
                total += 1
                if delta >= threshold:
                    break
        print('converged: {} delta after {} episodes'.format(delta, total))

    def run_leaves(self):
        """Take steps until board is terminal, add utility to values."""
        while len(self.values) < 136: # ~1000 episodes
            self.board.reset()
            while not self.board.is_terminal():
                action = self.random_policy()
                self.board.push(action)
            self.values[self.board] = self.board.utility()

    ## Save methods ##

    def get_values(self):
        return self.values.values

class RLSelfPlayTree(Tree):

    def change_values(self, values):
        self.table.values = values

    def get_best_actions(self, board):
        items = []
        best = max if board.turn() == 1 else min

        for action in board.get_actions():
            board.push(action)
            # unseen boards valued at zero (neutral)
            value = self.table.get(board, 0)
            items.append((action, value))
            board.pop()

        best_value = best(items, key=lambda x: x[1])[1]
        best_actions = [action for action,value in items if value==best_value]

        return best_actions

class MCSelfPlay(RLSelfPlay):

    def evaluate_episode(self):
        """Gradient descent. Shift values toward ACTUAL return reward.

        Undo board moves on each step. Reward is zero for non leaf nodes.
        Total reward G effectively decays by gamma each step."""
        G = self.board.utility()
        self.visits[self.board] += 1
        self.values[self.board] = G
        for _ in range(self.board.moves()):
            self.board.pop()
            self.visits[self.board] += 1
            delta = self.alpha * (G - self.values[self.board])
            self.values[self.board] += delta
            G *= self.gamma

    def episode_delta(self):
        """Return max absolute change in values. Don't change values."""
        G = self.board.utility()
        max_delta = G - self.values[self.board]

        for _ in range(self.board.moves()):
            self.board.pop()
            delta = self.alpha * (G - self.values[self.board])
            max_delta = max(abs(delta), max_delta)
            G *= self.gamma

        return max_delta

class TDSelfPlay(RLSelfPlay):

    def evaluate_episode(self):
        """Gradient descent. Shift values toward ESTIMATED return reward.

        Undo board moves on each step. Reward is zero for non leaf nodes.
        Reward G is effectively next step value decayed by gamma."""
        G = self.board.utility()
        self.visits[self.board] += 1
        self.values[self.board] = G
        for _ in range(self.board.moves()):
            self.board.pop()
            self.visits[self.board] += 1
            delta = self.alpha * (G - self.values[self.board])
            self.values[self.board] += delta
            # reward is 0 for non leaf nodes
            G = self.gamma * self.values[self.board]

    def episode_delta(self):
        """Return max absolute change in values. Don't change values."""
        G = self.board.utility()
        max_delta = G - self.values[self.board]

        for _ in range(self.board.moves()):
            self.board.pop()
            delta = self.alpha * (G - self.values[self.board])
            max_delta = max(abs(delta), max_delta)
            G = self.gamma * (self.values[self.board] + delta)

        return max_delta

class TDLSelfPlay(RLSelfPlay):

    def __init__(self, gamma=1, alpha=.5, epsilon=1, lambda_=.5,
                 values=None, visits=None):
        super().__init__(gamma, alpha, epsilon, values, visits)
        self.lambda_ = lambda_

    def evaluate_episode(self):
        """Gradient descent. Shift values toward AVERAGE of n-step TD rewards.

        Undo board moves on each step. Reward is zero for non leaf nodes.
        If lambda_ = 0, update is equivalent to TD(0)
            G = self.gamma * self.values[self.board]
        If lambda_ = 1, update is equivalent to MC
            G = self.gamma*G
        """
        G = self.board.utility()
        self.visits[self.board] += 1
        self.values[self.board] = G
        for _ in range(self.board.moves()):
            self.board.pop()
            self.visits[self.board] += 1
            delta = self.alpha * (G - self.values[self.board])
            self.values[self.board] += delta
            G = (1 - self.lambda_) * self.values[self.board] + self.lambda_*G
            G *= self.gamma

    def episode_delta(self):
        """Return max absolute change in values. Don't change values."""
        G = self.board.utility()
        max_delta = G - self.values[self.board]

        for _ in range(self.board.moves()):
            self.board.pop()
            delta = self.alpha * (G - self.values[self.board])
            max_delta = max(abs(delta), max_delta)
            G = (1 - self.lambda_) * (self.values[self.board] + delta) + (
                self.lambda_*G)
            G *= self.gamma

        return max_delta
