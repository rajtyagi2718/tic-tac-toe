import numpy as np

from board import Board
from transposition import Table
from search import RandomSearch, TreeSearch
from dp import UniformTree, DiscountTree
from minimax import MinimaxTree, NegaminTree
from rl import RLSelfPlayTree

import os

class Agent:
    """
    Agents have a name, use search to decide action during game run.

    Agents take action as a game runs. To decide which action, query search
    for policy. Most searches are tree searches that query a tree to evaluate
    possible actions. Results of previous games are maintained as a
    (win, loss, draw) record.
    """

    def __init__(self, name, search):
        self.name = name
        self.search = search
        self.record = np.zeros(3, int)

    def win_share(self):
        """Return float. Each game is worth 1 unit. +1 win, +.5 draw."""
        win,draw,_ = self.record
        return win + .5*draw

    def act(self, game):
        """Return action for game step."""
        return self.search.policy(game)

    def reset_record(self):
        """Set record to 0."""
        self.record[:] = 0

    def update_record(self, utility):
        """Increment record given utility from game. 1 win, -1 loss, 0 draw."""
        if not utility:
            self.record[1] += 1
        elif utility == 1:
            self.record[0] += 1
        else:
            self.record[2] += 1

    def __repr__(self):
        """Return str of name : record."""
        return self.name + ' : ' + str(self.record)

class Spawn:
    """
    Creates agent instances. Some predefined agent types stored by name (str).

    Agents are routinely created. Spawn stores common agent types by name for
    easy retrieval. Also allows for custom agents given other parameters.

    Search parameter is tree, tree parameter is kwargs. Attributes are dict
    maps from name (str) to either: search (cls), tree (cls or inst), tree kwargs (if tree is cls).
    """

    names = []
    searches = {}
    trees = {}
    tree_kwargs = {}

    @classmethod
    def get_agent(cls, name, search=None, tree=None, tree_kwargs=()):
        """Return instance of agent. If parameters null, use cls attr."""
        search = cls.get_search(name, search, tree, tree_kwargs)
        return Agent(name, search)

    @classmethod
    def get_search(cls, name, search, tree, tree_kwargs):
        """Return search instance from tree."""
        if not search:
            if name not in cls.names:
                name = 'user'
            search, tree, tree_kwargs = cls._get_search_kwargs(name)
        if tree_kwargs:
            tree = tree(**tree_kwargs)
        return search(tree)

    @classmethod
    def _get_search_kwargs(cls, name):
        """Return triplet (search, tree, kwargs) mapped from cls attributes."""
        return (cls.searches[name], cls.trees[name], cls.tree_kwargs[name])

    @classmethod
    def add_agent(cls, name, search, tree, tree_kwargs, set_tree):
        """Add predefined agent type to cls attributes for easy creation."""
        cls.names.append(name)
        cls.searches[name] = search
        if set_tree:
            tree = cls._get_tree(name, tree, tree_kwargs)
            tree_kwargs.clear()
        cls.trees[name] = tree
        cls.tree_kwargs[name] = tree_kwargs

    @classmethod
    def _get_tree(cls, name, tree, tree_kwargs):
        table = Table()
        loaded = table.load_values(name)
        if not loaded:
            if name in ('mc', 'tdl', 'qs', 'ts'):
                print('not loaded!', name)
                exit()
            table = None
        tree_kwargs['table'] = table
        tree = tree(**tree_kwargs)
        if not loaded:
            if name in ('mc', 'tdl', 'qs', 'ts'):
                print('not loaded!', name)
                exit()
            tree.table.save_values(name)
        return tree

Spawn.add_agent('random', RandomSearch, (), {}, False)

Spawn.add_agent('uniform', TreeSearch, UniformTree, {}, True)
Spawn.add_agent('discount', TreeSearch, DiscountTree, {}, True)
Spawn.add_agent('minimax', TreeSearch, MinimaxTree, {}, True)
Spawn.add_agent('negamin', TreeSearch, NegaminTree, {}, True)

Spawn.add_agent('mc', TreeSearch, RLSelfPlayTree, {}, True)
# Spawn.add_agent('td', TreeSearch, RLSelfPlayTree, {}, True)
Spawn.add_agent('tdl', TreeSearch, RLSelfPlayTree, {}, True)
Spawn.add_agent('qs', TreeSearch, RLSelfPlayTree, {}, True)
Spawn.add_agent('ts', TreeSearch, RLSelfPlayTree, {}, True)


# 'ts', 'mcts', 'ucts', 'user'
