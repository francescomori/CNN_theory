"""Helpers shared by the figure scripts: paths, command-line options, parallel map."""
import argparse, os, sys
from multiprocessing import Pool
import matplotlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'figures')


def parse(description):
    """Parse the options --quick, --replot and --workers N. Returns the argparse namespace.

    --quick uses fewer seeds or grid points and writes to figures/quick/ and data/*_quick.npz.
    """
    global OUT
    matplotlib.use('Agg')      # the renderer notebooks use too, so tight_layout is the same everywhere
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--quick', action='store_true', help='fewer seeds or grid points, for a fast check')
    p.add_argument('--replot', action='store_true', help='redraw from the cached data only')
    p.add_argument('--workers', type=int, default=None, help='parallel processes (default: all cores)')
    args = p.parse_args()
    if args.quick:
        OUT = os.path.join(ROOT, 'figures', 'quick')
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    return args


def pmap(fn, jobs, workers=None):
    """[fn(job) for job in jobs], computed by `workers` processes (all cores if None)."""
    if workers == 1:
        return [fn(j) for j in jobs]
    with Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=1)


def data_path(name, quick):
    """Path of the cached results `name` in data/ (a separate file for --quick runs)."""
    return os.path.join(DATA, f'{name}{"_quick" if quick else ""}.npz')


def save(fig, name):
    """Save the matplotlib figure as figures/<name>.pdf (figures/quick/ for --quick runs)."""
    fig.savefig(os.path.join(OUT, f'{name}.pdf'))


def show():
    """Display all open figures in a notebook, then close them."""
    import io
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    for n in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(n).savefig(buf, format='png', dpi=110, bbox_inches='tight')
        display(Image(buf.getvalue()))
    plt.close('all')
