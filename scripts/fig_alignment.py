"""Figure 4 (alignment-a.pdf, alignment-b.pdf): alignment gamma_s of the CNN kernel with
each zero-error solution s (model.alignment) against training time, across the plateau.

Main panels: one random initial condition at D = 1e12. Dotted line: specialization time.
Insets: fraction p_s of 4096 random initial conditions (same D) that end at solution s,
with binomial standard errors; dashed line: the uniform value. (a) k_s = 3, (b) k_s = 4.
All curves are ODE solutions. Parameters: g = erf, P = 6, k_t = 1, c = 0, eta = 0.5,
sigma = 0, T = 1.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common                # first: puts the repository root on sys.path for cnnlcn
from cnnlcn import model, solve
from cnnlcn.style import set_style, FIGSIZE, MARGINS

P, KT, KSS, C, T, ETA, SIGMA = 6, 1, (3, 4), 0.0, 1.0, 0.5, 0.0
D_ONE = 10**12
N_WIN = 4096
PLACE = {0: dict(color='C0', ls='-', marker='o'), 1: dict(color='C1', ls='--', marker='s'),
         2: dict(color='C2', ls=':', marker='^'), 3: dict(color='C4', ls='-.', marker='D')}


def gammas(a, Y):
    """Alignments gamma_s along a CNN trajectory Y (state size, n). Returns (n_solutions, n)."""
    return np.array([model.alignment(a, *model.unpack(a, 'cnn', y)) for y in Y.T]).T


def run(ks):
    """One trajectory at D_ONE (seed 0). Inputs: ks. Returns (alpha, gamma, alpha_spec)."""
    a = model.Arch(P, KT, ks, C, T)
    y0 = model.random_init(a, 'cnn', D_ONE, 0)
    r = solve.integrate(a, 'cnn', y0, ETA, SIGMA, 1e5, n_out=600, log_grid=False, stop_eps=1e-5)
    return r['alpha'], gammas(a, r['Y']), solve.spec_time(a, y0, ETA, SIGMA)


def winner(job):
    """Solution reached by one run, job = (ks, seed): argmax_s gamma_s once eps_g < 1e-4. Returns an int."""
    ks, seed = job
    a = model.Arch(P, KT, ks, C, T)
    r = solve.integrate(a, 'cnn', model.random_init(a, 'cnn', D_ONE, seed), ETA, SIGMA, 1e5,
                        n_out=2, log_grid=False, stop_eps=1e-4)
    return int(np.argmax(gammas(a, r['Y'])[:, -1]))


def compute(args):
    """Trajectories and selected solutions for both k_s. Returns a dict."""
    out = {}
    for ks in KSS:
        out[f'a_{ks}'], out[f'gamma_{ks}'], out[f'spec_{ks}'] = run(ks)
        jobs = [(ks, s) for s in range(64 if args.quick else N_WIN)]
        out[f'win_{ks}'] = np.array(common.pmap(winner, jobs, args.workers))
    return out


def plot(d):
    """Draw both panels with their insets from the output of compute (or its cached npz)."""
    set_style()
    for ks, tag in zip(KSS, 'ab'):
        ss = list(range(ks - KT + 1))
        fig, ax = plt.subplots(figsize=FIGSIZE); fig.subplots_adjust(**MARGINS)
        al, gamma = d[f'a_{ks}'], d[f'gamma_{ks}']
        for s in ss:
            ax.plot(al, gamma[s], **PLACE[s], lw=2.2, markevery=(8 * s, 60), markersize=7)
        ax.axvline(float(d[f'spec_{ks}']), color='0.5', ls=(0, (1, 3)), lw=1.4, zorder=0)
        ax.axhline(0.0, color='0.5', ls='--', lw=1.2, zorder=0)
        ax.set_xlim(0, al[-1]); ax.set_ylim(-0.25, 1.05)
        ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(['0.0', '0.5', '1.0'])
        ax.set_xlabel(r'Training time $\alpha$')
        if ks == KSS[0]:
            ax.set_ylabel(r'Alignment $\gamma_s$')
        win = d[f'win_{ks}']
        p = np.bincount(win, minlength=len(ss)) / win.size
        ins = ax.inset_axes([0.14, 0.66, 0.30, 0.27])
        ins.bar(ss, p, yerr=np.sqrt(p * (1 - p) / win.size), width=0.7, capsize=2,
                color=[PLACE[s]['color'] for s in ss], error_kw=dict(lw=1))
        ins.axhline(1 / len(ss), color='0.3', ls='--', lw=1)
        ins.set_xticks(ss); ins.set_yticks([0, 0.5]); ins.set_ylim(0, 1.15 * max(p.max(), 1 / len(ss)) + 0.02)
        ins.tick_params(labelsize=11)
        ins.set_xlabel(r'$s$', fontsize=12, labelpad=1); ins.set_ylabel(r'$p_s$', fontsize=12, labelpad=4)
        h = [Line2D([], [], **PLACE[s], lw=2.2, label=rf'$\gamma_{s}$') for s in ss]
        ax.legend(handles=h, frameon=False, loc='upper center',
                  bbox_to_anchor=(0.5, -0.20), ncol=len(ss), fontsize=13, handlelength=2.2, columnspacing=1.1)
        common.save(fig, f'alignment-{tag}')


def main(args):
    """Compute (unless args.replot) and cache the results, then draw the figure."""
    path = common.data_path('alignment', args.quick)
    if not args.replot:
        np.savez(path, **compute(args))
    plot(np.load(path))


if __name__ == '__main__':
    main(common.parse(__doc__))
