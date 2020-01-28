
import numpy as np
from game import Game
from agent import Spawn
from search import TreeSearch
from dp import DiscountTree

from board_hash import HashTable
from board import Board

from itertools import combinations as comb

if __name__ == '__main__':
    agents = [Spawn.get_agent(name) for name in Spawn.names]
    results = []
    for agent1, agent2 in comb(agents, 2):
        G = Game(agent1=agent1, agent2=agent2)
        G.compete(100)
    print('\n'.join(map(str, agents)))
