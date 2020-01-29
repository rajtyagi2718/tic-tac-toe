# Self-Play RL vs DP Agents in Tic-Tac-Toe
Self-Play Reinforcement Learning (RL) has shown remarkable results in a variety of games. For smaller games, Dynamic Programming (DP) and Minimax tree search is the most efficient method for creating perfect agents. However, for larger games, this relies on expert domain knowledge and hard-coded heuristics. Self-play RL has the potential of generalizing a single tabula rasa learning algorithm across all types of games.

We experiment with self-play RL in the simple game of Tic-Tac-Toe. After developing an efficient representation of the game, we implement each type of AI agent.  For DP, we implement 3 types: Uniform, Discount, Minimax. We also have a baseline random agent. These fixed policies are played against 4 self-play RL agents: Monte Carlo (MC), Temporal Difference Lambda (TD), Q Search (Q), and Tree Strap (TS).

During policy improvement over 1000 games, we measure the current performance of the RL agents by playing the DP agents 100 times. The learning curves for each RL agent is shown below. 

## Learning Curves
![alt text](https://github.com/rajtyagi2718/tic-tac-toe/blob/master/data/plots.svg)

## How to Play
Tic-Tac-Toe or Noughts & Crosses is a 2-player deterministic game played on a 3x3 board. On each turn a player chooses an empty cell. A player wins if they select three in a row (or column, diagonal). The game ends in a draw if all cells are filled without a winner.

## MDP
The game can be viewed as a Markov Decision Process (MDP) where each board is a state and each player can take an action to advance to an afterstate. As a graph, afterstates are children. The root node is the empty board, and the leaves are the terminal boards (win, lose, or draw).  An upper bound for the number of states is 3^9 = 19,683. The actual number of states is 5478. Accounting for symmetry and transpositions, we can narrow it down 765 states.

To develop different policies, we create a table mapping each state to a value. Terminal states are valued at (1, -1, 0) for (agent1 win, agent2 win, draw). During game play, the agent will choose an action with the best afterstate. So agent1 finds the maximal child value and agent2 the minimal.

## DP
DP recursively generates the complete game tree, then backs up the values from the leaves to the root. The Uniform agent assumes actions are chosen randomly. So parent values equal the average of its children. The Discount agent is similar, except values decay for each ply of the backup. The Minimax agent assumes the opponent plays optimally as well. Therefore, each parent value will equal the optimal child value.

Minimax is a perfect player. It never loses.

## RL
RL agents learn iteratively through experience. After each game, the agent is given a reward based on the outcome. Each state along the game path is updated toward the outcome by gradient descent. The self-play variant has one agent playing both sides.

MC updates toward the actual reward. TD uses estimated rewards stored in the table. Q Search is similar to Q Learning by updating toward the best estimated reward, however the estimate is the result of a minimax search up to a certain depth. TS is similar to Q, except the estimated reward is backed up to all interior nodes of the minimax tree search.

## Variance in Values
Each RL agent converges to the optimal policy within 2000 games, and weakly within 750 games. Note that MC values have little variance since there are no intermediate rewards. TD relies of estimated rewards, which compounds variance with each action. Q is off-policy and has the most variance. Estimated returns are a result a multi-ply search. The variance trades off with a steady learning rate.

## Tree Strap Performance
TS has minimal variance and outstanding learning rate. It converges weakly within 200 games. When experimenting with random values initalization, it often converges within 100 games. Where as Q only updates the root of each minimax search tree toward the result of the search, TS adjusts all interior nodes as well, thus fully utilizing each search. 

## Tree Strap Generalizability
Self-play RL shows promise for larger games. With feature extraction, function approximation, and Alpha Beta search of greater depth, TS can be extended to more complex games.
