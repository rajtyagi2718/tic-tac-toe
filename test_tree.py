import unittest
from unittest import TestCase

# 752 states, 627 non leaves
class TestBoard(TestCase):

    @classmethod
    def get_boards(cls, num=10):
        keys = list(range(9))
        result = [Board()]
        for _ in range(num):
            random.shuffle(keys)
            board = Board()
            for i in range(9):
                board.play_key(keys[i])
                b = Board(list(board.values), list(board.played_keys),
                          set(board.open_keys), board.winner,
                          board.hash_value)
                result.append(b)
                if board.winner:
                    break
        return result

    def test_init(self):
        board = Board()

    def test_play(self):
        values = [[0]*9, [1]+[0]*8, [1,2]+[0]*7, [1,2,0,1]+[0]*5,
             [1,2,2,1]+[0]*5, [1,2,2,1,0,0,1,0,0]]
        moves = [0,1,2,3,4,5]
        turns = [1,2,1,2,1,2]
        winners = [None]*5 +[1]
        keys = [0,1,3,2,6]
        open_keys = [set(range(9))]
        for k in keys:
            open_keys.append(set(open_keys[-1].difference({k})))

        board = Board()
        for i in range(6):
            if i:
                self.assertEqual(board.last_key(), keys[i-1])
                board.undo_last_key()
                self.assertEqual(board.hash_value, HashBoard.hash(board))
                self.assertEqual(board.values, values[i-1])
                self.assertEqual(board.moves(), i-1)
                self.assertEqual(board.played_keys, keys[:i-1])
                self.assertEqual(set(board.open_keys), open_keys[i-1])
                self.assertEqual(board.winner, winners[i-1])
                self.assertEqual(board.turn(), turns[i-1])
                self.assertEqual(board.turn(), 3-board.other())
                board.play_key(keys[i-1])
            self.assertEqual(board.hash_value, HashBoard.hash(board))
            self.assertEqual(board.values, values[i])
            self.assertEqual(board.moves(), i)
            self.assertEqual(board.played_keys, keys[:i])
            self.assertEqual(set(board.open_keys), open_keys[i])
            self.assertEqual(board.winner, winners[i])
            self.assertEqual(board.turn(), turns[i])
            self.assertEqual(board.turn(), 3-board.other())
            if i == 5:
                break
            board.play_key(keys[i])
        self.assertEqual(board.winner, winners[5])

    def test_play_undo(self):
        for board in self.get_boards(num=3):
            while board.winner is None:
                key = random.sample([k for k in range(9) if not
                                    board.values[k]], 1)[0]
                board.play_key(key)
                counter = [0, 0, 0]
                played_keys = set()
                open_keys = set()
                for i in range(9):
                    counter[board.values[i]] += 1
                    if board.values[i]:
                        played_keys.add(i)
                    else:
                        open_keys.add(i)
                self.assertEqual(played_keys, set(board.played_keys))
                self.assertEqual(open_keys, set(board.open_keys))
                self.assertLessEqual(abs(counter[2]-counter[1]), 1)

                for _ in range(4):
                    if not random.randrange(8) and board.moves():
                        board.undo_last_key()

class TestSimulation(TestCase):

    def test_simulation(self):
        for board in TestBoard.get_boards():
            Simulation(board)

class TestGame(TestCase):

    def test_game_1(self):
        g = Game()
        g.play()

    def test_game_2(self):
        b = Board()
        pk = [6,1,2,8]
        for i in (pk):
            b.play_key(i)
        g = Game(board=b)
        g.play()

    def test_game_3(self):
        b = Board()
        pk = [0,6,3,4,8,2]
        for i in (pk):
            b.play_key(i)
        g = Game(board=b)
        g.play()

    def test_many(self):
        for board in TestBoard.get_boards():
            g = Game(board)
            g.play()

class PlayerTest:

    names = {'random': False, 'uniform': False, 'distance': False,
             'minimax': False, 'negamax': False, 'alphabeta': False,
             'heuristic': False, 'moveorder': False, 'iterative': False,
             'montecarlo': False, 'confidence': True}

    args = {'random': ((),), 'uniform': ((),), 'distance': ((),),
            'minimax': ((),), 'negamax': ((),), 'alphabeta': ((),),
            'heuristic': tuple((i,) for i in range(1, 10)),
            'moveorder': tuple((i,) for i in range(1, 10)),
            'iterative': tuple((i,) for i in range(1, 10)),
            'montecarlo': ((10,), (100,), (1000,)),
            'confidence': ((10,), (100,), (1000,))}

    @classmethod
    def test_inst(cls, inst, name):
        for arg in cls.args[name]:
            inst.player_test(name, arg)

    @classmethod
    def test_players(cls, inst):
        for name in cls.names:
            if cls.names[name]:
                for arg in cls.args[name]:
                    inst.player_test(name, arg)
                    print(inst, name, arg)

class TestSpawn(TestCase):

    def test_players(self):
        PlayerTest.test_players(self)

    def player_test(self, name, strategy_args):
        Game(player1=Spawn.get_player(name, strategy_args))
        Game(player2=Spawn.get_player(name, strategy_args))
        Game(player1=Spawn.get_player(name, strategy_args),
             player2=Spawn.get_player(name, strategy_args))

class TestPlayer(TestCase):

    def test_players(self):
        PlayerTest.test_players(self)

    def player_test(self, name, strategy_args):
        self.player_num_test(name, strategy_args, True)
        self.player_num_test(name, strategy_args, None, True)
        self.player_num_test(name, strategy_args, True, True)

    def player_num_test(self, name, strategy_args, player1=None, player2=None):
        if player1:
            player1=Spawn.get_player(name, strategy_args)
        if player2:
            player2=Spawn.get_player(name, strategy_args)
        for board in TestBoard.get_boards():
            g = Game(board=board, player1=player1, player2=player2)
            g.play()

class TestSearchAnalysis(TestCase):

    def test_players(self):
        PlayerTest.test_players(self)

    def player_test(self, name, strategy_args):
        for board in TestBoard.get_boards():
            if board.winner is not None:
                continue
            sa = SearchAnalysis(name, strategy_args, board)
            sa.suggest()
            sa.get_most_valuable()
            sa.get_key_values()
            sa.get_norm_key_values()

class TestDrawNegamaxTree(TestCase):

    names = {'alphabeta': False, 'heuristic': False, 'moveorder': False,
             'iterative': False, 'montecarlo': False, 'confidence': False}

    args = {'alphabeta': ((),),
            'heuristic': tuple((i,) for i in range(4, 10)),
            'moveorder': tuple((i,) for i in range(4, 10)),
            'iterative': tuple((i,) for i in range(1, 10)),
            'montecarlo': tuple((i,) for i in [1000]),
            'confidence': tuple((i,) for i in [1000])}

    def test_players(self):
        for name in self.names:
            if self.names[name]:
                for arg in self.args[name]:
                    self.player_test(name, arg)

    def player_test(self, name, strategy_arg):
        self.competition_test(name, strategy_arg)
        self.suggest_test(name, strategy_arg)

    def competition_test(self, name, strategy_arg, n=10):
        result, comp, off = Competition.check_draw('negamax', name,
                                                   strat2=strategy_arg, n=n)
        match_str = 'negamax vs ' + name
        for arg in strategy_arg:
            match_str += ' ' + str(arg)
        print(match_str, comp)
        self.assertTrue(result, msg=(match_str, comp, off))

        result, comp, off = Competition.check_draw(name, 'negamax',
                                                   strat1=strategy_arg, n=n)
        match_str = name
        for arg in strategy_arg:
            match_str += ' ' + str(arg)
        match_str +=  ' vs negamax'
        print(match_str, comp)
        self.assertTrue(result, msg=(match_str, comp, off))

    def suggest_test(self, name, strategy_arg, n=100):
        for board in TestBoard.get_boards(num=n):
            if board.winner is not None:
                continue
            sa = SearchAnalysis(name, strategy_arg, board)
            key = sa.suggest()
            nega = SearchAnalysis('negamax', (), board).get_most_valuable()
            self.assertTrue(key in nega, msg=(board, key, nega, sa.get_key_values()))

class TestTimePlayer(TestCase):

    game_time = [5, 10, 20]

    names = {'idtime': False, 'ucttime': False}

    args = {'idtime': ((10,),), 'ucttime': ((10000,),)}

    def test_players(self):
        for t in self.game_time:
            for name in self.names:
                if self.names[name]:
                    for arg in self.args[name]:
                        self.player_num_test(name, arg, t, True)
                        self.player_num_test(name, arg, t, None, True)
                        self.player_num_test(name, arg, t, True, True)

    def player_num_test(self, name, strategy_args, t, p1=None, p2=None):
        if p1:
            p1=Spawn.get_player(name, strategy_args)
        if p2:
            p2=Spawn.get_player(name, strategy_args)
        for _ in range(3):
            g = TimeGame(player1=p1, player2=p2, t=t)
            g.play()


if __name__ == '__main__':
    unittest.main()
