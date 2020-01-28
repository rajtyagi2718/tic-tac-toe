import numpy as np
import os

DATA_PATH = os.getcwd() + '/data/'

def main():
    """Call explore on BoardHash to fill HashTable."""
    if load_hash_table():
        return
    else:
        fill_hash_table()
        save_hash_table()

def fill_hash_table():
    BoardHash.explore()
    HashTable.hash_keys[1:] = BoardHash.hash_keys.reshape((2,9))

def save_hash_table():
    for name in ('hash_values', 'win_values', 'hash_keys'):
        np.save(DATA_PATH + 'hash_table_' + name + '.npy',
                getattr(HashTable, name))

def load_hash_table():
    for name in ('hash_values', 'win_values', 'hash_keys'):
        try:
            data = np.load(DATA_PATH + 'hash_table_' + name + '.npy')
        except FileNotFoundError:
            return False
        setattr(HashTable, name, data)
    return True

class HashTable:
    """Map board hash value to symm value. Map symm value to win value.

    Board hash value equals sum of hash keys. Grouping boards by symmetries and
    transpositions, hash values are hashed further to a representative. These
    are mapped to a win value.

        hash_values = array maps all board hash value to a representative
        win_values = array maps hash representative to win value
        hash_keys = 3x9 array maps (key, turn) to unique value
    """

    hash_values = np.zeros(19556+1, dtype=np.uint16)
    hash_values[:] = 1024
    win_values = np.zeros(765, dtype=np.uint16)
    win_values[:] = 3
    hash_keys = np.zeros((3,9), dtype=np.uint16)

    @classmethod
    def get_hash_key(cls, turn, key):
        """Returns 3**key * turn. Unique to each turn, key pair."""
        return cls.hash_keys[turn, key]

    @classmethod
    def get_winner(cls, hash_value):
        """Returns win value if not equal to 3 i.e. not terminal."""
        result = cls.win_values[hash_value]
        return result if result != 3 else None

    @classmethod
    def get_hash(cls, hash_value):
        """Returns hash representative which groups symmetry, transposition."""
        return int(cls.hash_values[hash_value])

class BoardHash:
    """Explore bitboard to generate complete state space, fill HashTable.

    class attributes:
        board -- bitboards, 3x3 boolean array for each agent of keys played
        hash_keys -- 2x3x3 array maps (agent, key[0], key[1]) to hash key int
        open_keys -- set of index tuples of keys played by neither agent
        hashstop -- int counts number of unique boards explored
        turn -- int for current agent turn: 0 for agent1, 1 for agent 2
    """
    board = np.zeros((2,3,3), dtype=bool)

    hash_keys = np.zeros((2,3,3), dtype=np.uint16)
    hash_keys[0] = np.geomspace(1, 3**8, 9, dtype=np.uint16).reshape((3,3))
    hash_keys[1] = 2*hash_keys[0]

    open_keys = set((i,j) for i in range(3) for j in range(3))

    hashstop = 0
    turn = 0

    @classmethod
    def explore(cls):
        """Recursively search complete game state, depth first."""
        if cls.cutoff():
            return

        for key in tuple(cls.open_keys):
            cls.push(key)
            cls.explore()
            cls.pop(key)

    ## Board methods ##

    @classmethod
    def push(cls, key):
        """Play key on board."""
        cls.board[cls.turn, key[0], key[1]] = True
        cls.open_keys.remove(key)
        cls.switch_turn()

    @classmethod
    def pop(cls, key):
        """Undo key from board."""
        cls.switch_turn()
        cls.board[cls.turn, key[0], key[1]] = False
        cls.open_keys.add(key)

    @classmethod
    def switch_turn(cls):
        """Switch agent turn: 0 to 1, 1 to 0."""
        cls.turn ^= 1

    ## Cutoff methods ##

    @classmethod
    def cutoff(cls):
        """Return bool to end explore stack. Bookkeep hash, win value."""
        return (cls.cutoff_transposition() or cls.cutoff_draw() or
                cls.cutoff_win_value())

    @classmethod
    def cutoff_transposition(cls):
        """Return bool if board or symmetries prev explored. Else set hash."""
        symm = cls.get_symm_hash()
        if any(cls.is_transposition(s) for s in symm):
            return True

        # set hash of board, symmetries to hashtop
        for s in symm:
            cls.set_hash_value(s)
        cls.hashstop += 1

        return False

    @classmethod
    def cutoff_draw(cls):
        """Return bool if board is draw. Set win value to 0 if so."""
        if cls.is_draw():
            cls.set_win_value(0)
            return True
        return False

    @classmethod
    def cutoff_win_value(cls):
        """Return bool if board has winner. Set win value if so."""
        w = cls.get_winner()
        if w:
            cls.set_win_value(w)
            return True
        return False

    ## Hash methods ##

    @classmethod
    def get_symm_hash(cls):
        """Return set of hashes of symmetric boards (rotations, reflections)."""
        result = set()
        # rotations
        for i in range(4):
            result.add(cls.get_hash_value(np.rot90(cls.board, i, (1, 2))))
        # vertical, horizontal reflections
        for i in range(1, 3):
            result.add(cls.get_hash_value(np.flip(cls.board, i)))
        # diagonal reflection
        result.add(cls.get_hash_value(np.transpose(cls.board, (0, 2, 1))))
        # other diagonal reflection
        result.add(cls.get_hash_value(np.transpose(cls.board[:, ::-1, ::-1],
                                              (    0, 2, 1))))
        return result

    @classmethod
    def get_hash_value(cls, board):
        """Return hash value of given board."""
        return np.sum(board * cls.hash_keys)

    @classmethod
    def set_hash_value(cls, board):
        """Set hash value of of given board."""
        HashTable.hash_values[board] = cls.hashstop

    @classmethod
    def set_win_value(cls, win_value):
        HashTable.win_values[cls.hashstop-1] = win_value

    ## Cutoff helper methods ##

    @classmethod
    def is_transposition(cls, s):
        return HashTable.hash_values[s] != 1024

    @classmethod
    def is_draw(cls):
        return not cls.open_keys

    @classmethod
    def get_winner(cls):
        if len(cls.open_keys) > 4: # moves < 5
            return 0
        for i in range(2):
            if (any((cls.board[i,j,:].all() or cls.board[i,:,j].all())
                    for j in range(3)) or
                cls.board[i].diagonal().all() or
                np.flipud(cls.board[i]).diagonal().all()):
                return i+1
        return 0

main()
