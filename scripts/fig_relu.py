"""Figure 10, Appendix J (relu-floor.pdf): asymptotic error with the ReLU activation.

Online SGD from perfect recovery; eps_g (computed exactly from the weights) is averaged
over the second half of each run, alpha in [400, 800], then over 8 runs per value of c.
Error bars: standard errors over the runs.
Parameters: g(z) = max(0, z), P = 4, k_s = k_t = 2, eta = 0.1, sigma = 1, T = 1, D = 400.
"""
import numpy as np
import matplotlib.pyplot as plt
import common                # first: puts the repository root on sys.path for cnnlcn
from cnnlcn import model, sgd
from cnnlcn.style import set_style, STUDENT

P, K, T, D, ETA, SIGMA = 4, 2, 1.0, 400, 0.1, 1.0
CS = np.linspace(0.0, 0.95, 8)
ALPHA_MAX = 800.0


def compute(args):
    """Mean and standard error of eps_g / (eta sigma^2) per c and student. Returns a dict."""
    n_runs, amax = (3, 200.0) if args.quick else (8, ALPHA_MAX)
    cs = np.repeat(CS, n_runs)                 # every (c, run) pair is one chain of a single batch
    out = dict(cs=CS)
    for st, seed in (('cnn', 1), ('lcn', 2)):
        r = sgd.simulate(model.Arch(P, K, K, 0.0, T), st, D, ETA, SIGMA, amax, record_every=0.5,
                         n_chains=len(cs), seed=seed, init='perfect', act='relu', cs=cs)
        per_run = r['eps'][:, r['alpha'] >= amax / 2].mean(1).reshape(len(CS), n_runs) / (ETA * SIGMA ** 2)
        out[f'{st}_mean'], out[f'{st}_se'] = per_run.mean(1), per_run.std(1) / np.sqrt(n_runs)
    return out


def plot(d):
    """Draw the figure from the output of compute (or its cached npz)."""
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.155, top=0.97)
    for st in ('cnn', 'lcn'):
        S = STUDENT[st]
        ax.errorbar(d['cs'], d[f'{st}_mean'], yerr=d[f'{st}_se'], color=S['color'], ls=S['ls'],
                    marker=S['marker'], capsize=3, label=f'{st.upper()} (SGD)')
    ax.set_xlabel(r'Correlation $c$'); ax.set_ylabel(r'$\varepsilon^\infty/(\eta\sigma^2)$')
    ax.legend(frameon=False, loc='upper left')
    common.save(fig, 'relu-floor')


def main(args):
    """Compute (unless args.replot) and cache the results, then draw the figure."""
    path = common.data_path('relu', args.quick)
    if not args.replot:
        np.savez(path, **compute(args))
    plot(np.load(path))


if __name__ == '__main__':
    main(common.parse(__doc__))
