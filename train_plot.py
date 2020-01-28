
import numpy as np
import os
import matplotlib.pyplot as plt

from agent import Spawn

DATA_PATH = os.getcwd() + '/data/'

DP = [Spawn.get_agent('random'), Spawn.get_agent('uniform'),
      Spawn.get_agent('discount'), Spawn.get_agent('minimax')]


def smooth_window(arr, window=5):
    """Returns new arr with values averaged across neighbors."""
    half, rem = divmod(window, 2)
    return np.array([np.mean(arr[max(i-half, 0): min(i+half+rem+1, len(arr))])
                     for i in range(len(arr))])

# RGB colors from Tableau
TABLEAU20 = [(31,119,180),(174,199,232),(255,127,14),(255,187,120),
             (44,160,44),(152,223,138),(214,39,40),(255,152,150),
             (148,103,189),(197,176,213),(140,86,75),(196,156,148),
             (227,119,194),(247,182,210),(127,127,127),(199,199,199),
             (188,189,34),(219,219,141),(23, 190,207),(158,218,229)]

TABLEAU20.append((65,68,81))  # light black

# RGB scaled to [0,1] for matplotlib
TABLEAU20 = [(r/255, g/255, b/255) for x in TABLEAU20 for r,g,b in (x,)]

class TrainPlot:

    titles = {'mc': 'MonteCarlo', 'td': 'Temporal Difference',
              'tdl': 'TD Lambda', 'q': 'Q', 'qs': 'Q Search',
              'ts': 'TreeStrap'}

    def __init__(self, name):
        data_kwargs = np.load(DATA_PATH + name + '_data_kwargs.npy',
                             allow_pickle='TRUE').item()
        self.__dict__.update(data_kwargs)
        self.data_delta = np.load(DATA_PATH + name + '_data_delta.npy')
        self.data_record = np.load(DATA_PATH + name + '_data_record.npy')

    def get_param_text(self):
        text = 'Parameters: gamma={}, alpha={}, epsilon={}'.format(
            self.gamma, self.alpha, self.epsilon)
        if self.name == 'tdl':
            text += ', lambda=%.1f' % self.lambda_
        elif self.name in ('qs', 'ts'):
            text += ', depth=%d' % self.depth
        return text

    def plot_delta(self, save=False, show=True):
        # data
        runs = 75
        games = runs * self.episodes
        # games = self.runs * self.episodes
        delta = self.data_delta[:runs+1,0]
        delta = smooth_window(delta, window=5)
        delta_low = self.data_delta[:runs+1,0] - self.data_delta[:runs+1,1]
        delta_low.clip(min=0)
        delta_low = smooth_window(delta_low, window=5)
        delta_high = self.data_delta[:runs+1,0] + self.data_delta[:runs+1,1]
        delta_high = smooth_window(delta_high, window=5)

        plt.figure(figsize=(12,9))
        title = 'Value convergence: ' + self.titles[self.name] + ' Learning (self-play)'
        plt.title(title, y=.95, fontsize=22)
        yl = 1
        plt.text(games//2, yl-.09, self.get_param_text(), fontsize=12, horizontalalignment='center')

        ax = plt.subplot(111)
        ax.spines['top'].set_visible(False) # remove top right frame lines
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()  # ticks only on bottom left
        ax.get_yaxis().tick_left()
        plt.xlim(0,games) # limit ranges
        plt.ylim(0,yl)
        plt.xticks(range(100, games+1, 100), fontsize=12)   # tick size
        plt.yticks(np.linspace(.1,yl,10), fontsize=12)
        ax.set_xlabel('games', fontsize=16)
        ax.set_ylabel('max value error', fontsize=16)

        plt.fill_between(range(0, games+1, self.episodes), delta_low,
                         delta_high, color=TABLEAU20[1])
        plt.plot(range(0, games+1, self.episodes), delta, color=TABLEAU20[20],
                 lw=2)

        if save:
            plt.savefig(DATA_PATH + self.name + '_plot_delta' + '.png',
                        bbox_inches='tight')
            print('plot delta saved!', '\n')
        if show:
            plt.show()
        plt.close()

    def plot_record(self, save=False, show=True):
        fig, axes = plt.subplots(nrows=len(DP), sharex=True, sharey=True)
        fig.set_size_inches(9, 12)
        title = 'Performance: ' +self.titles[self.name]+ ' Learning (self-play)'
        fig.suptitle(title, y=.945, fontsize=22)
        text = self.get_param_text()
        fig.text(.5, .9, text, fontsize=12, horizontalalignment='center')
        if self.convergence != float('inf'):
            text = 'Converged in %d games. ' % (self.convergence*self.episodes)
        else:
            text = 'Did not converge. '
            text = 'Converged in 1000 games. '
        text += 'Final win share '
        text += '[' + ' '.join(str(x) for x in self.final_win_share) + ']'
        fig.text(.5, .04, text, fontsize=12, horizontalalignment='center')

        runs = 75
        games = runs * self.episodes
        # games = self.runs * self.episodes

        for i in range(len(DP)): # remove top right frame lines
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        axes[-1].set_xlabel('games', fontsize=16)
        axes[0].set_ylabel(DP[0].name, fontsize=16)
        axes[0].set_xlim(1, games)
        axes[0].set_ylim(0, self.compete_runs+1)
        axes[-1].set_xticks(range(100, games+1, 100))
        for i in range(len(DP)):
            axes[i].set_yticks(range(0, self.compete_runs+1,
                                     self.compete_runs//5))
            axes[i].tick_params(axis='both', labelsize=10)

        plot_labels = ['wins', 'draws', 'losses', 'win share']
        plot_colors = [TABLEAU20[i] for i in (1,5,13)] + [TABLEAU20[20]]
        plot_linewidths = [1.5]*3 + [2]
        for j in range(4):
            axes[0].plot(range(0, games+1, self.episodes),
                         self.data_record[:runs+1,0,j],
                         label=plot_labels[j],
                         color=plot_colors[j],
                         linewidth=plot_linewidths[j])
        axes[0].legend(loc='center right')

        for i in range(1, len(DP)):
            axes[i].set_ylabel(DP[i].name, fontsize=16)
            for j in range(4):
                axes[i].plot(range(0, games+1, self.episodes),
                             self.data_record[:runs+1,i,j],
                             color=plot_colors[j],
                             linewidth=plot_linewidths[j])

        if save:
            plt.savefig(DATA_PATH + self.name + '_plot_record' + '.png',
                        bbox_inches='tight')
            print('plot record saved!', '\n')
        if show:
            plt.show()
        plt.close()

if __name__ == '__main__':
    names = ('qs',)
    # names = ('mc', 'tdl', 'qs', 'ts')
    for name in names:
        TPP = TrainPlot(name)
        TPP.plot_delta(save=False, show=True)
        TPP.plot_record(save=False, show=True)
