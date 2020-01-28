import numpy as np
from board_hash import HashTable

class Board:
    """
    Represents physical playing area: 3 by 3 grid.

    Enumerate cell positions as keys of flattened 1-d array of length 9.

        [ 0 1 2 ]
        [ 3 4 5 ]    -->    [0, 1, 2, 3, 4, 5, 6, 7, 8]
        [ 6 7 8 ]

    agent 1 assigned either X's or O's, agent 2 the other. During agents
    alternate placing respective pieces onto open position of board. So each
    cell containes either nothing, an X, or an O. Store a value for each key,
    indicating state of that position.

        0 -- open    1 -- occupied by agent 1    2 -- occupied by agent 2

    So current board state completely abstracted as a 1-d array of length 9,
    containing integers 0, 1, 2.  For example,

        [ _ _ _ ]
        [ _ _ _ ]    -->    [0, 0, 0, 0, 0, 0, 0, 0, 0] -- an empty board
        [ _ _ _ ]

        [ X _ O ]
        [ _ X _ ]    -->    [1, 0, 2, 0, 1, 0, 0, 0, 2] -- after 4 moves
        [ _ _ O ]

    This array is values attribute. All other attributes allow for efficient
    search of previous and following game states, used by AI agents in search.

    Attributes:
        values -- array of current board state
        played_keys -- list of cell positions played
        open_keys -- set of cell positions yet to be played
        winner --
            None -- no winner, board is not full
            0 -- draw, no winner    1 -- agent 1 won    2 -- agent 2 won
        hash_value -- hash value of current board
    """

    def __init__(self, values=None, played_keys=None, open_keys=None,
                 winner=None, hash_value=None):
        if values is None:
            values = np.zeros(9, int)
            played_keys = []
            open_keys = set(range(9))
            winner = None
            hash_value = 0
        self.values = values
        self.played_keys = played_keys
        self.open_keys = open_keys
        self.winner = winner
        self.hash_value = hash_value

    ## State methods: abstractions from values ##

    def moves(self):
        """Return number of keys already played."""
        return len(self.played_keys)

    def turn(self):
        """Return number of current agent.

        Current agent is next to act on board. agent 1 is first to act on
        empty board. agent 2 follows. Turns are alternated.
        """
        return 1 + self.moves() % 2

    def other(self):
        """Return opponents agent number i.e. the agent that last played."""
        return 2 - self.moves() % 2

    def last_key(self):
        """Return last played key or None if empty board."""
        try:
            return self.played_keys[-1]
        except IndexError:
            return None

    def is_terminal(self):
        """Return True if there is a winner or draw. Else play may continue."""
        return self.winner is not None

    def __hash__(self):
        """Return hash int. Equates symmetric boards, transpositions."""
        return HashTable.get_hash(self.hash_value)

    def __eq__(self, other):
        """Boards of equal hash are equal."""
        return hash(self) == hash(other) if isinstance(other, Board) else False

    ## Play methods: used during game runs and AI search, alter state ###

    def get_actions(self):
        """Return tuple of legal actions by agents. Actions are open keys."""
        return tuple(self.open_keys)

    def push(self, key):
        """Play key: set values to current agent number at key index."""
        assert not self.values[key], (key, self.values)
        self.values[key] = self.turn()
        # Turn is function of number of moves i.e. len of played_keys.
        # Call hash key before incrementing number of moves.
        self.hash_value += HashTable.get_hash_key(self.turn(), key)
        self.played_keys.append(key)
        self.open_keys.remove(key)
        self.winner = HashTable.get_winner(hash(self))

    def pop(self):
        """Undo play of last key. Return last key."""
        last_key = self.played_keys.pop()
        self.values[last_key] = 0
        self.open_keys.add(last_key)
        self.winner = None
        self.hash_value -= HashTable.get_hash_key(self.turn(), last_key)
        return last_key

    def reset(self):
        """Empty board. Ready for new game."""
        self.values[:] = 0
        self.played_keys.clear()
        self.open_keys = set(range(9))
        self.winner = None
        self.hash_value = 0

    ## Search methods: used by AI agents ##

    def utility(self):
        """Return state value from agent1 pov: +1 if win, -1 lose, 0 else."""
        if self.winner == 1:
            return 1
        if self.winner == 2:
            return -1
        return 0

    ## Other methods ##

    def __repr__(self):
        """Return string representation of board as matrix, minimal notation.

        Examples:
            [0 0 0]    [0 1 0]    [1 1 2]
            [0 0 0]    [2 2 0]    [2 2 1]
            [0 0 0]    [1 2 1]    [1 2 1]
        """
        return '\n'.join('[' + ' '.join(map(str, self.values[i:i+3])) + ']'
                         for i in range(0,9,3))


    def __str__(self):
        """Return user friendly string representation of board.

        Examples:
            ///////////    ///////////     ///////////
            // . . . //    // . x . //     // x o . //
            // . . . //    // o o . //     // o o x //
            // . . . //    // . x x //     // o x x //
            ///////////    ///////////     ///////////
        """
        result = '/'*11 + '\n'
        for i in range(0, 9, 3):
            row = '// '
            row += ' '.join(self.key_to_piece(key) for key in range(i, i+3))
            row += ' //'
            result += row + '\n'
        result += '/'*11
        return result

    def key_to_piece(self, key):
        """Return x o or . if key is played by agent 1, 2, or unplayed."""
        value = self.values[key]
        if not value:
            return '.'
        if value == 1:
            return 'x'
        return 'o'

    def copy(self):
        """Return deep copy of current instance."""
        return Board(np.array(self.values), list(self.played_keys),
                     set(self.open_keys), self.winner, self.hash_value)
