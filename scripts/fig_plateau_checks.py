"""Figure 7, Appendix E (spec-app-a.pdf, spec-app-b.pdf): specialization time of the CNN
against D in two further settings, as in Figure 3(b), with fits a + b ln D.

(a) k_t = 2, k_s = 2, 3, 4, c = 0, eta = 0.5.
(b) k_t = 1, k_s = 1, 2, 3, c = 0 (faded) and c = 0.5, eta = 0.2.
Mean and standard error over 128 random initial conditions; ODE solutions.
Parameters: g = erf, P = 6, sigma = 0, T = 1, gamma_th = 0.99, D from 1e6 to 1e10.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common                # first: puts the repository root on sys.path for cnnlcn
from cnnlcn import model, solve
from cnnlcn.style import set_style, FIGSIZE, MARGINS
from fig_plateau import draw_times, ks_legend

P, T, SIGMA = 6, 1.0, 0.0
DS = list(np.logspace(6, 10, 9))
N_SEEDS = 128
SETUPS = {'a': dict(kt=2, kss=(2, 3, 4), cs=(0.0,), eta=0.5),
          'b': dict(kt=1, kss=(1, 2, 3), cs=(0.0, 0.5), eta=0.2)}


def spec(job):
    """Specialization time of one run, job = (kt, ks, c, eta, D, seed). Returns a float."""
    kt, ks, c, eta, D, seed = job
    a = model.Arch(P, kt, ks, c, T)
    return solve.spec_time(a, model.random_init(a, 'cnn', D, seed), eta, SIGMA)


def compute(args):
    """Times for both panels, each (n_ks, n_c, n_D, n_seeds). Returns a dict with a, b and Ds."""
    seeds = range(16 if args.quick else N_SEEDS)
    out = {'Ds': np.array(DS)}
    for tag, s in SETUPS.items():
        jobs = [(s['kt'], ks, c, s['eta'], D, sd) for ks in s['kss'] for c in s['cs'] for D in DS for sd in seeds]
        out[tag] = np.array(common.pmap(spec, jobs, args.workers)).reshape(
            len(s['kss']), len(s['cs']), len(DS), len(seeds))
    return out


def plot(d):
    """Draw both panels from the output of compute (or its cached npz)."""
    set_style()
    for tag, s in SETUPS.items():
        fig, ax = plt.subplots(figsize=FIGSIZE); fig.subplots_adjust(**MARGINS)
        for i, ks in enumerate(s['kss']):
            for k, c in enumerate(s['cs']):
                faded = len(s['cs']) > 1 and c == 0.0
                draw_times(ax, d['Ds'], d[tag][i, k], ks, alpha=0.35 if faded else 1.0)
        ax.set_xscale('log')
        ax.set_xlabel(r'Feature dimension $D$')
        ax.set_ylabel(r'Specialization time $\alpha_{\rm spec}$', fontsize=19)
        extra = []
        if len(s['cs']) > 1:
            extra = [Line2D([], [], color='0.3', alpha=0.35, lw=2.4, label=r'$c = 0$'),
                     Line2D([], [], color='0.3', lw=2.4, label=r'$c = 0.5$')]
        ks_legend(ax, s['kss'], extra)
        common.save(fig, f'spec-app-{tag}')


def main(args):
    """Compute (unless args.replot) and cache the results, then draw the figure."""
    path = common.data_path('plateau_checks', args.quick)
    if not args.replot:
        np.savez(path, **compute(args))
    plot(np.load(path))


if __name__ == '__main__':
    main(common.parse(__doc__))
