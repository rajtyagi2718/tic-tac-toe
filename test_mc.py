import unittest

import numpy as np

from mc import MonteCarlo

class TestPolicy(unittest.TestCase):

    def test_policy(self):
        MC = MonteCarlo()
        keys = np.zeros(9, int)
        keys = np.zeros(9, int)
        for _ in range(10**3):
            keys[MC.policy()] += 1
        print('random policy?', keys)

        keys[:] = 0
        MC.board.push(4)
        state = hash(MC.board)
        MC.values[state] = -10
        MC.board.pop()

        for _ in range(10**3):
            keys[MC.policy()] += 1
        self.assertEqual(keys[4], 0)
        print('policy null 4', keys)

    def test_eg_policy(self):
        MC = MonteCarlo()
        keys = np.zeros(9, int)
        for _ in range(10**3):
            keys[MC.epsilon_greedy_policy(hash(MC.board))] += 1
        print('eg policy random?', keys)

        MC.visits[hash(MC.board)] = MC.epsilon_constant
        keys[:] = 0
        MC.board.push(1)
        MC.values[hash(MC.board)] = 10
        MC.board.pop()

        for _ in range(10**4):
            keys[MC.epsilon_greedy_policy(hash(MC.board))] += 1
        print('eg policy tripled 1,3,5,7?', keys)

class TestGenerateEpisode(unittest.TestCase):

    def test_generate_episode(self):
        MC = MonteCarlo()
        paths = []
        for _ in range(10):
            states = MC.generate_episode()
            self.assertLess(states[-1], hash(MC.board))
            paths.append(states)
        print('sample state paths:', paths)

class TestEvaluateEpisode(unittest.TestCase):

    def test_evaluate_episode(self):
        MC = MonteCarlo()
        for _ in range(10):
            states = MC.generate_episode()
            pre_visits = np.array([MC.visits[s] for s in [0] + states])
            pre_values = np.array([MC.values[s] for s in [0] + states])
            MC.evaluate_episode(states)
            post_visits = np.array([MC.visits[s] for s in [0] + states])
            post_values = np.array([MC.values[s] for s in [0] + states])
            diff_values = pre_values - post_values
            diff_visits = pre_visits - post_visits
            print('sample path value deltas:', diff_values)
            self.assertTrue(np.all(diff_visits) == 1)

    def test_evaluate_episode_delta(self):
        MC = MonteCarlo()
        deltas = []
        for _ in range(100):
            deltas.append(MC.evaluate_episode_delta(MC.generate_episode()))
            self.assertTrue(-1 <= deltas[-1] <= 1)
        self.assertTrue(any(d for d in deltas))
        print('sample path value max deltas:', deltas)


class TestRun(unittest.TestCase):

    def test_run_cutoff(self):
        MC = MonteCarlo()
        MC.run(max_episodes=1000, threshold=.01)

    def test_run(self):
        MC = MonteCarlo()
        MC.run(max_episodes=10**5, threshold=.01, interval=100, checks=30)


if __name__ == '__main__':
    unittest.main()
