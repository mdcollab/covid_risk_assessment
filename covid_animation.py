
import numpy as np
import matplotlib.colors as ListedColormap
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

plt.style.use('seaborn')

COLOR_MAPPINGS = {
    'S': 'gold',
    'I': 'crimson',
    'R': 'lightseagreen', 
    'QS': 'gold',
    'QI': 'crimson',
    'QR': 'lightseagreen', 
}

######################################
# Animation Funcs ####################
######################################

def animate(day_logs, ax):
    N = len(day_logs)
    x = np.array(day_logs.index)
    y = np.array(
        [1 if state in ('QI', 'QR', 'QS') else 2 for state in day_logs]
    )

    for state, color in COLOR_MAPPINGS.items():
        plot_x = [v for i, v in enumerate(x) if day_logs[i] == state]
        plot_y = [v for i, v in enumerate(y) if day_logs[i] == state]
        scat = ax.scatter(plot_x, plot_y, color=color)

    return scat,