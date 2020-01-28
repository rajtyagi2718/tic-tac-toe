##############
## Preamble ##
##############

from board import Board
from agent import Spawn

class Game:
    """
    Handles game play: two agents alternate turns on board until win or draw.

    A board is given with two agents. During a run, they alternate taking a
    step by pushing an action onto the board. First agent starts on an empty
    board. When the board is terminal, the game run ends with a winner and
    loser, or both draw.
    """

    def __init__(self, board=None, agent1=None, agent2=None):
        if board is None:
            board = Board()
        if agent1 is None:
            agent1 = Spawn.get_agent('random')
        if agent2 is None:
            agent2 = Spawn.get_agent('random')
        self.board = board
        self.agent1 = agent1
        self.agent2 = agent2

    def actions(self):
        """Return list of legal actions i.e. open keys on board."""
        return self.board.get_actions()

    def current_agent(self):
        """Return agent for current turn. Game awaits their action."""
        if self.board.turn() == 1:
            return self.agent1
        return self.agent2

    def other_agent(self):
        """Current agent acts next. Return the other agent."""
        if self.board.other() == 1:
            return self.agent1
        return self.agent2

    def step(self):
        """Query current agent to act. Push action onto board."""
        action = self.current_agent().act(self)
        self.board.push(action)

    def run(self):
        """Take steps until board is terminal. Return winner: 0, 1, or 2."""
        while not self.board.is_terminal():
            self.step()
        self.update_agent_records()
        return self.board.winner

    def update_agent_records(self):
        """Increment each agent's (win, draw, loss) records."""
        u = self.board.utility()
        self.agent1.update_record(u)
        self.agent2.update_record(-u)

    def reset(self):
        """Return to intial state. Ready for new run."""
        self.board.reset()

    def change_agents(self, agent1=None, agent2=None):
        if agent1 is not None:
            self.agent1 = agent1
        if agent2 is not None:
            self.agent2 = agent2

    def swap_agents(self):
        self.agent1, self.agent2 = self.agent2, self.agent1

    def runs(self, num_runs):
        for _ in range(num_runs):
            self.reset()
            self.run()

    def compete(self, num_runs):
        """Run game num_runs times. Swap who goes first halfway through."""
        m = num_runs // 2
        self.runs(m)
        self.swap_agents()
        self.runs(num_runs-m)
