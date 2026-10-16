"""Figure 3 (plateau-a.pdf, plateau-b.pdf): plateau and specialization time of the CNN.

(a) eps_g against alpha at k_s = 1, 2, 3, one random initial condition at D = 1e12.
(b) specialization time alpha_spec (solve.spec_time) against D, mean and standard error
    over 128 random initial conditions, with fits a + b ln D.
All curves are ODE solutions from random initial conditions (student weights iid N(0, 1)).
Parameters: g = erf, P = 6, k_t = 1, c = 0, eta = 0.5, sigma = 0, T = 1, gamma_th = 0.99
(the default of solve.spec_time).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common                # first: puts the repository root on sys.path for cnnlcn
from cnnlcn import model, solve
from cnnlcn.style import set_style, KS, FIGSIZE, MARGINS, row_major

P, KT, KSS, C, T, ETA, SIGMA = 6, 1, (1, 2, 3), 0.0, 1.0, 0.5, 0.0
D_A = 10**12                                   # dimension of the panel (a) runs
DS = list(np.logspace(6, 12, 13))
N_SEEDS = 128


def spec(job):
    """Specialization time of one run, job = (ks, D, seed). Returns a float (nan if none by 1e5)."""
    ks, D, seed = job
    a = model.Arch(P, KT, ks, C, T)
    return solve.spec_time(a, model.random_init(a, 'cnn', D, seed), ETA, SIGMA)


def trajectory(ks):
    """eps_g(alpha) of one run at D_A until eps_g < 1e-4. Inputs: ks. Returns (alpha, eps)."""
    a = model.Arch(P, KT, ks, C, T)
    r = solve.integrate(a, 'cnn', model.random_init(a, 'cnn', D_A, 0), ETA, SIGMA, 1e5,
                        n_out=400, stop_eps=1e-4)
    return r['alpha'], r['eps']


def compute(args):
    """Panel (a) curves and panel (b) times (array (len(KSS), len(DS), n_seeds)). Returns a dict."""
    seeds = range(16 if args.quick else N_SEEDS)
    out = dict(Ds=np.array(DS))
    for ks, (al, ep) in zip(KSS, common.pmap(trajectory, list(KSS), args.workers)):
        out[f'a_{ks}'], out[f'e_{ks}'] = al, ep
    jobs = [(ks, D, s) for ks in KSS for D in DS for s in seeds]
    out['spec'] = np.array(common.pmap(spec, jobs, args.workers)).reshape(len(KSS), len(DS), len(seeds))
    return out


def draw_times(ax, Ds, tim, ks, alpha=1.0):
    """Plot mean and standard error of tim (n_D, n_seeds) against Ds with the fit a + b ln D.

    Inputs : ax, Ds (n_D,), tim (nan for runs that never specialized), ks (sets the style),
             alpha (opacity).
    """
    y = np.nanmean(tim, -1)
    se = np.nanstd(tim, -1) / np.sqrt(np.sum(~np.isnan(tim), -1))
    b, a = np.polyfit(np.log(Ds), y, 1)
    ax.errorbar(Ds, y, yerr=se, color=KS[ks]['color'], ls='none', marker=KS[ks]['marker'],
                markersize=7, capsize=3, alpha=alpha)
    xx = np.logspace(np.log10(Ds[0]), np.log10(Ds[-1]), 50)
    ax.plot(xx, a + b * np.log(xx), color=KS[ks]['color'], ls=KS[ks]['ls'], lw=2.0, alpha=alpha)


def ks_legend(ax, kss, extra=()):
    """Legend below the axes with one entry per k_s in kss, plus the handles in extra."""
    h = [Line2D([], [], lw=2.2, markersize=7, label=rf'$k_s = {k}$', **KS[k]) for k in kss] + list(extra)
    ax.legend(handles=row_major(h, 3), frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.20),
              ncol=3, fontsize=13, handlelength=2.2, columnspacing=1.1)


def plot(d):
    """Draw both panels from the output of compute (or its cached npz)."""
    set_style()
    fig, ax = plt.subplots(figsize=FIGSIZE); fig.subplots_adjust(**MARGINS)
    for ks in KSS:
        al, ep = d[f'a_{ks}'], d[f'e_{ks}']
        ax.plot(al[al > 0], ep[al > 0], lw=2.2, markevery=40, markersize=7, **KS[ks])
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(3e-4, 5)
    ax.set_xlabel(r'Training time $\alpha$'); ax.set_ylabel(r'Generalization error $\varepsilon_g$')
    ks_legend(ax, KSS)
    common.save(fig, 'plateau-a')

    fig, ax = plt.subplots(figsize=FIGSIZE); fig.subplots_adjust(**MARGINS)
    for i, ks in enumerate(KSS):
        draw_times(ax, d['Ds'], d['spec'][i], ks)
    ax.set_xscale('log')
    ax.set_xlabel(r'Feature dimension $D$')
    ax.set_ylabel(r'Specialization time $\alpha_{\rm spec}$', fontsize=19)
    ks_legend(ax, KSS)
    common.save(fig, 'plateau-b')


def main(args):
    """Compute (unless args.replot) and cache the results, then draw the figure."""
    path = common.data_path('plateau', args.quick)
    if not args.replot:
        np.savez(path, **compute(args))
    plot(np.load(path))


if __name__ == '__main__':
    main(common.parse(__doc__))
