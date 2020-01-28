
from tree import Tree, NegamaxTree

eval_max = 130

class AlphaBetaTree(NegamaxTree):

    # value, exact_flag, best

    def __init__(self):
        Tree.__init__(self)

    def explore(self, board, alpha=-1, beta=1):
        """Add children in depth first procedure. Bookkeep keys, board,
        hash incrementally during visit. Backtrack actions postvisit.
        Check transposition table during expansion."""
        result, item = self.cutoff_test(board, beta)
        if result:
            return item[0]

        # set starting value below alpha cutoff, best key won't remain None
        value = -2
        best = None

        for key in board.open_keys:
            board.push(key)
            child = -self.explore(board, -beta, -alpha)
            if value < child:
                value = child
                best = key
            board.pop()

            if value >= beta:
                self.table[board] = value, False, best
                return value
            alpha = max(value, alpha)

        self.table[board] = value, True, best
        return value

    def cutoff_test(self, board, beta):
        """End explore recursion if board is terminal, depth is reached, or
        board is transposition or symmetric. Return boolean and item."""
        item = self.table[board]
        result, item = self.table_test(board, beta, item)
        if result:
            return True, item
        result, item = self.terminal_test(board, item)
        if result:
            return True, item
        return False, item

    def table_test(self, board, beta, item):
        if item is not None:
            value, exact_flag, _ = item
            if exact_flag or value >= beta:
                return True, item
            else:
                del self.table[board]
        return False, item

    def terminal_test(self, board, item):
        if not board.is_terminal():
            return False, item
        item = (self.get_utility(board), True, None)
        self.table[board] = item
        return True, item

    def most_valuable(self, board):
        perm, item = self.table.get_perm_item(board)
        best = item[2]
        assert perm[best] in board.open_keys, (perm, item, board)
        return [perm[best]]

class HeuristicTree(AlphaBetaTree):

    # value, exact_flag, depth, best, e_val

    def explore(self, board, depth, alpha=-eval_max, beta=eval_max):
        """Add children in depth first procedure. Bookkeep keys, board,
        hash incrementally during visit. Backtrack actions postvisit.
        Check transposition table during expansion."""
        result, item = self.cutoff_test(board, depth, beta)
        if result:
            return item[0]

        # set starting value below alpha cutoff, best key won't remain None
        value = -eval_max-1
        best = None
        e_val = item[4] if item is not None else None

        for key in board.open_keys:
            board.push(key)
            child = -self.explore(board, depth-1, -beta, -alpha)
            if value < child:
                value = child
                best = key
            board.pop()

            if value >= beta:
                self.table[board] = (value, False, depth, best, e_val)
                return value
            alpha = max(value, alpha)

        self.table[board] = (value, True, depth, best, e_val)
        return value

    def cutoff_test(self, board, depth, beta):
        """End explore recursion if board is terminal, depth is reached, or
        board is transposition or symmetric. Return boolean and item."""
        item = self.table[board]
        result, item = self.table_test(board, depth, beta, item)
        if result:
            return True, item
        result, item = self.terminal_test(board, item)
        if result:
            return True, item
        result, item = self.depth_test(board, depth, item)
        if result:
            return True, item
        return False, item

    def table_test(self, board, depth, beta, item):
        if item is not None:
            value, exact_flag, prev_depth, _, _ = item
            if prev_depth >= depth and (exact_flag or value >= beta):
                return True, item
            else:
                del self.table[board]
        return False, item

    def terminal_test(self, board, item):
        if not board.is_terminal():
            return False, item
        item = (self.get_utility(board), True, 10, None, None)
        self.table[board] = item
        return True, item

    def depth_test(self, board, depth, item):
        if depth:
            return False, item
        e_val = item[4] if item is not None else board.evaluation()
        # if e_val is still None, then item set in table from previous game
        if e_val is None:
            e_val = board.evaluation()
        self.table[board] = item = (e_val, False, depth, None, e_val)
        return True, item

    def get_utility(self, board):
        """Return 0 for draw, -eval_max for either player win. Pov of board turn."""
        return abs(board.utility())*-eval_max

    def most_valuable(self, board):
        perm, item = self.table.get_perm_item(board)
        best = item[3]
        return [perm[best]]

    def norm_value(self, board, value):
        """Return value linearly scaled from 0 to 100. From point of view of
        current player at given parent board. POV always opponents, so always
        switch."""
        return .5 - value/(2*eval_max)

class MoveOrderTree(HeuristicTree):

    # value, exact_flag, depth, best, e_val

    def explore(self, board, depth, alpha=-eval_max, beta=eval_max):
        """Add children in depth first procedure. Bookkeep keys, board,
        hash incrementally during visit. Backtrack actions postvisit.
        Check transposition table during expansion."""
        result, item = self.cutoff_test(board, depth, beta)
        if result:
            return item[0]

        # set starting value below alpha cutoff, best key won't remain None
        value = -eval_max-1
        best = None
        e_val = item[4] if item is not None else None

        open_keys = sorted(board.open_keys,
                           key=lambda k: self.get_evaluation(board, k))
        for key in open_keys:
            board.push(key)
            child = -self.explore(board, depth-1, -beta, -alpha)
            if value < child:
                value = child
                best = key
            board.pop()

            if value >= beta:
                self.table[board] = (value, False, depth, best, e_val)
                return value
            alpha = max(value, alpha)

        self.table[board] = (value, True, depth, best, e_val)
        return value

    def get_evaluation(self, board, key):
        board.push(key)
        item = self.table[board]
        if item is not None:
            e_val = item[4] if item[4] is not None else item[0]
        else:
            e_val = board.evaluation()
            self.table[board] = (e_val, False, 0, None, e_val)
        board.pop()
        return e_val

class IterativeDeepeningTree(MoveOrderTree):

    # value, exact_flag, depth, best, e_val

    def principal_explore(self, board, depth, alpha=-eval_max, beta=eval_max):
        result, item, perm = self.principal_cutoff_test(board, depth, beta)
        if result:
            return item[0]

        # set starting value below alpha cutoff, best key won't remain None
        value = -eval_max-1
        best = None
        e_val = item[4] if item is not None else None

        # if item[3] is None, item set in previous game
        if depth > 1 and item[3] is not None:
            principal = perm[item[3]]
            board.push(principal)
            child = -self.principal_explore(board, depth-1, -beta, -alpha)
            if value < child:
                value = child
                best = principal
            board.pop()

            if value >= beta:
                self.table[board] = (value, False, depth, best, e_val)
                return value
            alpha = max(value, alpha)

            open_keys = list(board.open_keys.difference((principal,)))
        else:
            open_keys = list(board.open_keys)

        open_keys.sort(key=lambda k: self.get_evaluation(board, k))

        for key in open_keys:
            board.push(key)
            child = -self.explore(board, depth-1, -beta, -alpha)
            if value < child:
                value = child
                best = key
            board.pop()

            if value >= beta:
                self.table[board] = (value, False, depth, best, e_val)
                return value
            alpha = max(value, alpha)

        self.table[board] = (value, True, depth, best, e_val)
        return value

    def principal_cutoff_test(self, board, depth, beta):
        """End explore recursion if board is terminal, depth is reached, or
        board is transposition or symmetric. Return boolean and item."""
        permitem = self.table.get_perm_item(board)
        if permitem is not None:
            perm, item = permitem
        else:
            perm = item = None
        result, item = self.table_test(board, depth, beta, item)
        if result:
            return True, item, perm
        result, item = self.terminal_test(board, item)
        if result:
            return True, item, perm
        result, item = self.depth_test(board, depth, item)
        if result:
            return True, item, perm
        return False, item, perm

class TimeIterativeDeepeningTree(IterativeDeepeningTree):

    def principal_explore(self, alarm, board, depth, alpha=-eval_max, beta=eval_max):
        result, item, symm = self.principal_cutoff_test(alarm, board, depth, beta)
        if result:
            return item[0]

        # set starting value below alpha cutoff, best key won't remain None
        value = -eval_max-1
        best = None
        e_val = item[4] if item is not None else None

        if depth > 1:
            principal = item[3] if not symm else board.symm_col(item[3])
            board.push(principal)
            child = -self.principal_explore(alarm, board, depth-1, -beta, -alpha)
            if value < child:
                value = child
                best = principal
            board.pop()

            if value >= beta:
                self.table[board] = (value, False, depth, best, e_val)
                return value
            alpha = max(value, alpha)

            open_keys = list(board.open_keys.difference((principal,)))
        else:
            open_keys = list(board.open_keys)

        open_keys.sort(key=lambda k: self.get_evaluation(board, k))

        for key in open_keys:
            board.push(key)
            child = -self.explore(board, depth-1, -beta, -alpha)
            if value < child:
                value = child
                best = key
            board.pop()

            if value >= beta:
                self.table[board] = (value, False, depth, best, e_val)
                return value
            alpha = max(value, alpha)

        self.table[board] = (value, True, depth, best, e_val)
        return value

    def principal_cutoff_test(self, alarm, board, depth, beta):
        """End explore recursion if board is terminal, depth is reached, or
        board is transposition or symmetric. Return boolean and item."""
        symmitem = self.table.get_symm_item(board)
        if symmitem is not None:
            symm, item = symmitem
        else:
            symm = item = None
        if self.alarm_test(alarm):
            return True, item
        result, item = self.table_test(board, depth, beta, item)
        if result:
            return True, item, symm
        result, item = self.terminal_test(board, item)
        if result:
            return True, item, symm
        result, item = self.depth_test(board, depth, item)
        if result:
            return True, item, symm
        return False, item, symm

    def explore(self, alarm, board, depth, alpha=-eval_max, beta=eval_max):
        """Add children in depth first procedure. Bookkeep keys, board,
        hash incrementally during visit. Backtrack actions postvisit.
        Check transposition table during expansion."""
        result, item = self.cutoff_test(alarm, board, depth, beta)
        if result:
            return item[0]

        # set starting value below alpha cutoff, best key won't remain None
        value = -eval_max-1
        best = None
        e_val = item[4] if item is not None else None

        open_keys = sorted(board.open_keys,
                           key=lambda k: self.get_evaluation(board, k))
        for key in open_keys:
            board.push(key)
            child = -self.explore(alarm, board, depth-1, -beta, -alpha)
            if value < child:
                value = child
                best = key
            board.pop()

            if value >= beta:
                self.table[board] = (value, False, depth, best, e_val)
                return value
            alpha = max(value, alpha)

        self.table[board] = (value, True, depth, best, e_val)
        return value

    def cutoff_test(self, alarm, board, depth, beta):
        """End explore recursion if board is terminal, depth is reached, or
        board is transposition or symmetric. Return boolean and item."""
        item = self.table[board]
        if self.alarm_test(alarm):
            return True, item
        result, item = self.table_test(board, depth, beta, item)
        if result:
            return True, item
        result, item = self.terminal_test(board, item)
        if result:
            return True, item
        result, item = self.depth_test(board, depth, item)
        if result:
            return True, item
        return False, item

    def alarm_test(self, alarm):
        return time.time() > alarm
