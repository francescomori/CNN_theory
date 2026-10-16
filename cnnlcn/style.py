"""Figure style shared by the figure scripts."""
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, NullFormatter, FixedLocator

FIGSIZE = (5.6, 4.9)                                          # one panel
MARGINS = dict(left=0.19, right=0.965, bottom=0.32, top=0.965)  # its axes, with room for a legend below
STUDENT = {'cnn': dict(color='C0', ls='-', marker='o'),
           'lcn': dict(color='C3', ls='--', marker='s')}
KS = {1: dict(color='C0', ls='-', marker='o'), 2: dict(color='C3', ls='--', marker='s'),   # by kernel size
      3: dict(color='C2', ls=':', marker='^'), 4: dict(color='C4', ls='-.', marker='D')}


def set_style():
    """Set the matplotlib rcParams of the paper's figures (large fonts, no grid)."""
    mpl.rcParams.update({
        'font.size': 17, 'axes.labelsize': 21, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
        'legend.fontsize': 15, 'lines.linewidth': 2.2, 'lines.markersize': 8,
        'axes.grid': False, 'savefig.bbox': None,
    })


def plain_log(ax, axis, ticks):
    """Label a log axis with plain decimals at the given ticks.

    Inputs : ax (matplotlib Axes), axis ('x' or 'y'), ticks (list of tick values).
    """
    a = ax.yaxis if axis == 'y' else ax.xaxis
    a.set_major_locator(FixedLocator(ticks))
    f = ScalarFormatter()
    f.set_scientific(False)
    a.set_major_formatter(f)
    a.set_minor_formatter(NullFormatter())


def row_major(handles, ncol):
    """Reorder legend handles so that a legend with ncol columns reads row by row. Returns a list."""
    rows = -(-len(handles) // ncol)
    return [handles[r * ncol + j] for j in range(ncol) for r in range(rows) if r * ncol + j < len(handles)]
