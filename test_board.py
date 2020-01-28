from board import Board, HashBoard
import random
import unittest

class TestBoard(unittest.TestCase):

    @classmethod
    def get_boards(cls, num=10):
        keys = list(range(9))
        result = [Board()]
        for _ in range(num):
            random.shuffle(keys)
            board = Board()
            for i in range(9):
                board.push(keys[i])
                b = Board(list(board.values), list(board.played_keys),
                          set(board.open_keys), board.winner,
                          list(board.hash_values))
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
                board.pop()
                self.assertEqual(board.hash_values[0], HashBoard.hash(board))
                self.assertEqual(board.values, values[i-1])
                self.assertEqual(board.moves(), i-1)
                self.assertEqual(board.played_keys, keys[:i-1])
                self.assertEqual(set(board.open_keys), open_keys[i-1])
                self.assertEqual(board.winner, winners[i-1])
                self.assertEqual(board.turn(), turns[i-1])
                self.assertEqual(board.turn(), 3-board.other())
                board.push(keys[i-1])
            self.assertEqual(board.hash_values[0], HashBoard.hash(board))
            self.assertEqual(board.values, values[i])
            self.assertEqual(board.moves(), i)
            self.assertEqual(board.played_keys, keys[:i])
            self.assertEqual(set(board.open_keys), open_keys[i])
            self.assertEqual(board.winner, winners[i])
            self.assertEqual(board.turn(), turns[i])
            self.assertEqual(board.turn(), 3-board.other())
            if i == 5:
                break
            board.push(keys[i])
        self.assertEqual(board.winner, winners[5])

    def test_play_undo(self):
        for board in self.get_boards(num=3):
            while board.winner is None:
                key = random.sample([k for k in range(9) if not
                                    board.values[k]], 1)[0]
                board.push(key)
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
                        board.pop()

if __name__ == '__main__':
    unittest.main()
