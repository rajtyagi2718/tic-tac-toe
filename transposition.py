import numpy as np
import os

DATA_PATH = os.getcwd() + '/data/'

class Table:
    """
    Array maps board by hash value. No reference to board object is kept.

    Use board hash value as unique key. Transpositions, symmetric boards
    collide in hash function, saving computation and memory.
    """

    def __init__(self, values=None):
        self.default = 3.14159265
        if values is None:
            values = np.empty(765)
            values[:] = self.default
        self.values = values

    def save_values(self, name):
        np.save(DATA_PATH + name + '_data_values.npy', self.values)

    def load_values(self, name):
        try:
            self.values = np.load(DATA_PATH + name + '_data_values.npy')
            return True
        except FileNotFoundError:
            return False

    def __setitem__(self, board, item):
        self.values[hash(board)] = item

    def __getitem__(self, board):
        result = self.values[hash(board)]
        if result != self.default:
            return result
        raise KeyError(board)

    def __delitem__(self, board):
        self.values[hash(board)] = self.default

    def clear(self):
        self.values[:] = self.default

    def __contains__(self, board):
        return self.values[hash(board)] != self.default

    def get(self, board, default=None):
        try:
            return self[board]
        except KeyError:
            return default

class DefaultTable(Table):
    """Default variant of Table.

    Values attribute remains same. Methods allow defaultdict functionality on
    top of standard dict.
    """

    def __init__(self, values=None, default_fcn=int):
        super().__init__(values)
        self.default_fcn = default_fcn

    def __getitem__(self, board):
        result = self.values[hash(board)]
        if result != self.default:
            return result
        return self.__missing__(board)

    def __missing__(self, board):
        self[board] = result = self.default_fcn()
        return result

    def get(self, board, default=None):
        result = self.values[hash(board)]
        if result != self.default:
            return result
        return default

class Set:
    """
    Set stores boards by hash value. No reference to board object is kept.

    Use board hash value as unique key. Transpositions, symmetric boards
    collide in hash function, saving computation and memory.
    """

    def __init__(self, values=None):
        if values is None:
            values = set()
        self.values = values

    def __len__(self):
        return len(self.values)

    def add(self, board):
        self.values.add(hash(board))

    def clear(self):
        self.values.clear()

    def __contains__(self, board):
        return hash(board) in self.values


# class Table:
#     """
#     Dict maps board by hash value. No reference to board object is kept.
#
#     Use board hash value as unique key. Transpositions, symmetric boards
#     collide in hash function, saving computation and memory.
#     """
#
#     def __init__(self, values=None):
#         if values is None:
#             values = {}
#         self.values = values
#
#     def __len__(self):
#         return len(self.values)
#
#     def __setitem__(self, board, item):
#         self.values[hash(board)] = item
#
#     def __getitem__(self, board):
#         return self.values[hash(board)]
#
#     def __delitem__(self, board):
#         del self.values[hash(board)]
#
#     def clear(self):
#         self.values.clear()
#
#     def __contains__(self, board):
#         return hash(board) in self.values
#
#     def get(self, board, default=None):
#         try:
#             return self[board]
#         except KeyError:
#             return default
#
# class DefaultTable(Table):
#     """Default dict variant of Table.
#
#     Values attribute remains same. Methods allow defaultdict functionality on
#     top of standard dict.
#     """
#
#     def __init__(self, values=None, default_fcn=int):
#         super().__init__(values)
#         self.default_fcn = default_fcn
#
#     def __getitem__(self, board):
#         try:
#             return self.values[hash(board)]
#         except KeyError:
#             return self.__missing__(board)
#
#     def __missing__(self, board):
#         self[board] = result = self.default_fcn()
#         return result
#
#     def get(self, board, default=None):
#         try:
#             return self.values[hash(board)]
#         except KeyError:
#             return default
#
# class Set:
#     """
#     Set stores boards by hash value. No reference to board object is kept.
#
#     Use board hash value as unique key. Transpositions, symmetric boards
#     collide in hash function, saving computation and memory.
#     """
#
#     def __init__(self, values=None):
#         if values is None:
#             values = set()
#         self.values = values
#
#     def __len__(self):
#         return len(self.values)
#
#     def add(self, board):
#         self.values.add(hash(board))
#
#     def clear(self):
#         self.values.clear()
#
#     def __contains__(self, board):
#         return hash(board) in self.values
