
import numpy as np
from itertools import product

from game import Game
from agent import Spawn
from search import TreeSearch
from transposition import Table
from rl import RLSelfPlayTree, MCSelfPlay, TDSelfPlay, TDLSelfPlay
from ql import QSelfPlay, QSSelfPlay
from ts import TSSelfPlay

import os

DATA_PATH = os.getcwd() + '/data/'

DP = [Spawn.get_agent('random'), Spawn.get_agent('uniform'),
      Spawn.get_agent('discount'), Spawn.get_agent('minimax')]

class Train:

    def __init__(self, rl, name, episodes, runs, compete_runs):
        self.rl = rl
        self.name = name
        self.episodes = episodes
        self.runs = runs
        self.compete_runs = compete_runs
        self.rl_agent = Spawn.get_agent(name + '_play', TreeSearch,
                                        RLSelfPlayTree,
                                        {'table':Table(rl.get_values())})
        self.convergence = None
        self.start_win_share = None
        self.final_win_share = None
        self.data_delta = np.zeros((runs+1, 2))
        # wins draws loss win_share
        self.data_record = np.zeros((runs+1, len(DP), 4), int)
        self.game = Game()

    def run(self):
        self.add_delta(0)
        self.add_data_record(0)
        self.set_start_win_share()
        for i in range(1, self.runs+1):
            self.rl.run(self.episodes)
            self.add_delta(i)
            self.add_data_record(i)
        self.set_convergence()
        self.set_final_win_share()

    def add_delta(self, i):
        deltas = np.array([self.rl.get_episode_delta() for i in range(10)])
        self.data_delta[i] = [np.mean(deltas), np.std(deltas)]

    def add_data_record(self, i):
        self.rl_agent.search.tree.change_values(self.rl.get_values())
        for j in range(self.data_record.shape[1]):
            self.rl_agent.reset_record()
            DP[j].reset_record()
            self.game.change_agents(agent1=self.rl_agent, agent2=DP[j])
            self.game.compete(self.compete_runs)
            self.data_record[i,j,:3] = self.rl_agent.record
            self.data_record[i,j,3] = self.rl_agent.win_share()

    def set_convergence(self):
        i = self.runs-1
        while self.convergence_test(i):
            i -= 1
        i += 1
        if i == self.runs:
            i = float('inf')
        self.convergence = i

    def convergence_test(self, i):
        """Return True if no losses against all DP agents during run i."""
        return not self.data_record[i,:,2].any()

    def set_final_win_share(self):
        self.final_win_share = self.data_record[-1,:,-1]

    def set_start_win_share(self):
        self.start_win_share = self.data_record[0,:,-1]

    def get_data_kwargs(self):
        result = {key:value for key,value in self.__dict__.items()
                  if key in ('name', 'episodes', 'runs', 'compete_runs',
                             'convergence', 'start_win_share',
                             'final_win_share')}
        for param in ('alpha', 'gamma', 'epsilon'):
            result[param] = getattr(self.rl, param)
        if self.name == 'tdl':
            result['lambda_'] = self.rl.lambda_
        elif self.name in ('qs', 'ts'):
            result['depth'] = self.rl.depth
        return result

    def save_data(self):
        np.save(DATA_PATH + self.name + '_data_values.npy', self.rl.get_values())
        np.save(DATA_PATH + self.name + '_data_kwargs.npy',
                self.get_data_kwargs())
        np.save(DATA_PATH + self.name + '_data_delta.npy', self.data_delta)
        np.save(DATA_PATH + self.name + '_data_record.npy', self.data_record)
        print('data saved!', '\n')


def win_share_gt(a, b):
    return max((a,b), key=lambda x: list(reversed(x))) is a

def tune_param(rl, name, episodes, runs, compete_runs,
               gammas=None, alphas=None, epsilons=None, lambdas_=None,
               depths=None):
    print(name)
    print('episodes=%d, runs=%d, complete_runs=%d' %
          (episodes, runs, compete_runs))
    games = episodes * runs
    if gammas is None:
        gammas = (1, .9)
    if alphas is None:
        alphas = iter([.1,.3,.5,.7])
    if epsilons is None:
        epsilons = (1,5,25)
    if name == 'tdl':
        if lambdas_ is None:
            lambdas_ = iter([0,.2,.5])
    elif name in ('qs', 'ts'):
        if depths is None:
            depths = range(1, 3)
    parameters = (gammas, alphas, epsilons, lambdas_, depths)
    parameters = product(*filter(None, parameters))

    try:
        rl_data_kwargs = np.load(DATA_PATH + name + '_data_kwargs.npy',
                                allow_pickle='TRUE').item()
        min_convergence = rl_data_kwargs['convergence']
        max_win_share = rl_data_kwargs['final_win_share']
    except FileNotFoundError:
        min_convergence = float('inf')
        max_win_share = [-float('inf')]*4

    for param in parameters:
        print('param:', *param)
        rlselfplay = rl(*param)
        TP = Train(rlselfplay, name, episodes, runs, compete_runs)
        TP.run()
        convergence = TP.convergence
        win_share = TP.final_win_share
        print('convergence:', convergence)
        print('start_win_share:', TP.start_win_share)
        print('final_win_share:', win_share)
        print()
        if convergence < min_convergence:
            min_convergence = convergence
            TP.save_data()
        elif min_convergence == float('inf'):
            if win_share_gt(win_share, max_win_share):
                max_win_share = win_share
                TP.save_data()

if __name__ == '__main__':
    tune_param(TDSelfPlay, 'td', 10, 100, 100, gammas=[1])
