import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mc import MCSelfPlay
# from td import TDSelfPlay
from td import TDSelfPlay
from ql import QLSelfPlay
from dp import UniformTree
from minimax import MinimaxTree

UT = UniformTree()
UNIFORM_ITEMS = UT.table.table.items()
UNIFORM_VALUES = np.array([x[1] for x in sorted(UNIFORM_ITEMS,
                           key=lambda x: x[0])])

MT = MinimaxTree()
MINIMAX_ITEMS = MT.table.table.items()
MINIMAX_VALUES = np.array([x[1] for x in sorted(MINIMAX_ITEMS,
                           key=lambda x: x[0])])


class LearnValues:

    def __init__(self, rl, episodes=100, runs=100, items=MINIMAX_ITEMS):
        self.rl = rl
        self.episodes = episodes
        self.runs = runs
        self.items = items
        self.values = self.get_sorted_values(items)
        self.data = np.zeros((runs+1, len(self.values)))

    def run(self):
        self.add_data(0)
        for i in range(1, self.runs+1):
            self.rl.run_episodes(self.episodes)
            self.add_data(i)

    def add_data(self, i):
        self.data[i] = self.get_learned_values() - self.values

    def get_learned_values(self):
        if len(self.rl.values) == len(self.values):
            rl_items = self.rl.values.items()
        else:
            rl_items = self.complete_values()
        return self.get_sorted_values(rl_items)

    def get_sorted_values(self, items):
        return np.array([x[1] for x in sorted(items, key=lambda x: x[0])])

    def complete_values(self):
        return ((k, self.rl.values.get(k, 2+v)) for k,v in self.items)

    ## Analysis of Error

    def plot_central_states(self):
        # ax[i][j].errorbar(x, y, e, capsize=2, linestyle='-',
                          # marker='o', color=colors[i])

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('runs (1000 episodes)') # x == x0
        ax.set_ylabel('mean value error') # y == x2
        abs_data = np.abs(self.data)
        ax.errorbar(range(self.data.shape[0]),
                    np.mean(abs_data, axis=1),
                    np.std(abs_data, axis=1))
        plt.show()

    def plot_last_run(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('states') # x == x1
        ax.set_ylabel('value error') # y == x2
        ax.scatter(range(self.data.shape[1]), self.data[-1])
        plt.show()

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_ylabel('runs (1000 episodes)') # y == x0
        ax.set_xlabel('states') # x == x1
        ax.set_zlabel('value error')
        x0,x1 = self.data.shape
        X1,X0 = np.meshgrid(range(x1), range(x0))
        ax.plot_wireframe(X1, X0, self.data)
        plt.show()

if __name__ == '__main__':
    episodes = 100
    runs = 300
    total = episodes*runs
    for gamma in (i/10 for i in range(10, 9, -1)):
        LV = LearnValues(QLSelfPlay(epsilon_constant=total//100), episodes, runs)
        LV.run()
        # LV = LearnValues(TDLambdaSelfPlay(epsilon_constant=total//10), episodes, runs)
        # LV.run()
        # LV = LearnValues(TDSelfPlay(epsilon_constant=total//10), episodes, runs)
        # LV.run()
        # LV = LearnValues(MCSelfPlay(epsilon_constant=total//10), episodes, runs)
        # LV.run()
        print('gamma:', gamma)
        LV.plot_central_states()
        LV.plot_last_run()
        # LV.plot()
