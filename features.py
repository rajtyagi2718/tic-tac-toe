import numpy as np

from board import Board
from board_hash import HashTable
from board_slices import SLICES, WINNER_SLICES

class Features:
    """Extract binary features from board.

    Reduce the state space to a number of binary features that are turned on
    when the board has a particular characteristic.

    attributes:
        methods -- (method, feature length) of each type of feature
    """

    methods = [('is_terminal', 3), ('is_terminal_next', 3), ('is_trap', 2),
               ('get_incomplete', 2)]

    @classmethod
    def get_features(cls, board):
        """Return boolean array of binary features."""
        result = [0]*10
        r = 0
        for method, length in cls.methods:
            feat = getattr(cls, method)(board)
            if any(feat):
                result[r:r+length] = feat
                break
            r +=+ length

        return np.array(result)

    @classmethod
    def is_terminal(cls, board):
        """Return boelean triplet of (is_draw, is_agent1_win, is_agent2_win)."""
        result = [0]*3
        winner = HashTable.get_winner(hash(board))
        if winner is not None:
            result[winner] = 1
        return result

    @classmethod
    def is_terminal_next(cls, board):
        """Return tuple of whether current agent can win with next move."""
        result = [0]*3
        draw = (board.moves() == 8)
        for hsh in cls.get_next_hash(board):
            winner = HashTable.get_winner(hsh)
            if winner:
                result[winner] = 1
                return result
            draw &= (winner is not None)
        result[0] = int(draw)
        return result

    @classmethod
    def is_trap(cls, board):
        """Return tuple of whether current agent can force win in two turns."""
        result = [0,0]
        if board.moves() < 4:
            return result

        t = board.turn()
        for action in board.get_actions():
            singles = False
            for slc in WINNER_SLICES[action]:
                if (slc[0] ^ slc[1]) and (
                    board.values[slc[0]] == t or board.values[slc[1]] == t):
                    if not singles:
                        singles = True
                    else:
                        result[t-1] = 1
                        return result

        return result

    @classmethod
    def get_incomplete(cls, board):
        """Extract tuple of incomplete, potentially winning subslices.

        Sublice example:
            [ X _ O ]    (0,3) -- double for X
            [ X _ O ]    (0,8) -- double for X
            [ _ O X ]    (2)   -- single for O
                         (7)   -- single for O
            [ 0 1 2 ]
            [ 3 4 5 ]    Position indices for reference.
            [ 6 7 8 ]

        Count number of singles and doubles for each agent. Return tuple of
        difference of agent1 and agent2.
        """
        result = [0,0]  # [singles, doubles]
        counter = np.zeros(3, int)

        for slc in SLICES:
            counter[:] = 0
            for i in slc:
                counter[board.values[i]] += 1
            if counter[1] and not counter[2]:
                result[counter[1]-1] += 1
            elif counter[2] and not counter[1]:
                result[counter[2]-1] -= 1

        return result

    @classmethod
    def get_next_hash(cls, board):
        """Generator of hash of all afterstates."""
        for action in board.get_actions():
            board.push(action)
            result = hash(board)
            board.pop()
            yield result
