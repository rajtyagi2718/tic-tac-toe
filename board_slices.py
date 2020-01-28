"""
SLICES lists all possible winning triplets. This determines winner on board.

Agent wins if places three pieces in a line. There are 3 types of slices, total of 8. If any one contains all same pieces, that's a win.

    [ X X _ ]
    [ O O O ]  --  O's win along the second row
    [ _ _ X ]

    [ X _ X ]
    [ O O X ]  --  X's win along the third column
    [ O _ X ]

    [ X _ O ]
    [ O X _ ]  --  X's win along the first diagonal
    [ _ _ X ]
"""

SLICES = [] # flattened indices of each slice of board
SLICES.extend(tuple(range(i, i+3)) for i in range(0, 9, 3)) # rows
SLICES.extend(tuple(range(i,9,3)) for i in range(3)) # columns
SLICES.append(tuple(range(0,9,4))) # diagonal (main)
SLICES.append(tuple(range(2,8,2))) # diagonal (other)
SLICES = tuple(SLICES)

"""
WINNER_SLICES maps keys (k) list of pairs (i,j). Each triplet (i,j,k) is slice.

Game checks for winner after position is played. Check only relevant slices.
Any one position may intersect with 2 to 4 different slices. Note board stores
last played position. So only check intersecting slices, and for each slice
check other 2 positions against last played position.

    [(X)_ O ]    If position 0 is last played. Check three slices: first row,
    [ _ X _ ]    first column, first diagonal. Of those check pairs: (1,2),
    [ _ _ O ]    (3,6), (4,8) against position 0 for equality.

    [ 0 1 2 ]
    [ 3 4 5 ]    Position indices for reference.
    [ 6 7 8 ]
"""

WINNER_SLICES = [[] for _ in range(9)]
for s in SLICES:
    for i in range(3):
        WINNER_SLICES[s[i]].append(s[:i] + s[i+1:])
for i in range(9):
    WINNER_SLICES[i] = tuple(WINNER_SLICES[i])
WINNER_SLICES = tuple(WINNER_SLICES)
