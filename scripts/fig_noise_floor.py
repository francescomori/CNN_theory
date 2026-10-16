"""Figure 5 (noise-floor-a/b/c.pdf): asymptotic error eps_inf / (eta sigma^2).

Lines: the small-eta closed forms, Eqs. (eps-inf-A) and (eps-inf-B). Markers: fixed point
of the ODEs at finite eta (solve.noise_floor).
(a) against c at P = 6, (b) against P at c = 0, (c) against P at c = 0.5.
Parameters: g = erf, k_s = k_t = 2, eta = 2e-3, sigma = 1, T = 1.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common                # first: puts the repository root on sys.path for cnnlcn
from cnnlcn import model, solve, closed_forms as cf
from cnnlcn.style import set_style, STUDENT, FIGSIZE, MARGINS, plain_log

K, T, ETA, SIGMA = 2, 1.0, 2e-3, 1.0
P_A = 6                                  # P of panel (a)
PS = [3, 4, 5, 6, 8]
CS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95]


def floors(job):
    """ODE floors / (eta sigma^2) of the CNN and the LCN at job = (P, c). Returns a pair."""
    P, c = job
    a = model.Arch(P, K, K, c, T)
    return tuple(solve.noise_floor(a, st, ETA, SIGMA) / (ETA * SIGMA ** 2) for st in ('cnn', 'lcn'))


def compute(args):
    """Floors for the three panels. Returns a dict with cs, Ps and a, b, c (n, 2)."""
    cs = CS[::2] if args.quick else CS
    jobs = [(P_A, c) for c in cs] + [(P, 0.0) for P in PS] + [(P, 0.5) for P in PS]
    res = np.array(common.pmap(floors, jobs, args.workers))
    n = len(cs)
    return dict(cs=np.array(cs), Ps=np.array(PS), a=res[:n], b=res[n:n + len(PS)], c=res[n + len(PS):])


def panel(x, y, xa, ya_cnn, ya_lcn, xlabel, name, logxy):
    """Draw one panel and save it as figures/<name>.pdf.

    Inputs : x (n,), y (n, 2) ODE floors of the CNN and the LCN (markers); xa, ya_cnn, ya_lcn
             closed forms (lines); xlabel; name; logxy (log axes, for the panels against P).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINS)
    A, B = STUDENT['cnn'], STUDENT['lcn']
    ax.plot(xa, ya_cnn, color=A['color'], ls=A['ls'], lw=2.0)
    ax.plot(xa, ya_lcn, color=B['color'], ls=B['ls'], lw=2.0)
    ax.plot(x, y[:, 0], ls='none', marker=A['marker'], color=A['color'])
    ax.plot(x, y[:, 1], ls='none', marker=B['marker'], color=B['color'])
    if logxy:
        ax.set_xscale('log'); ax.set_yscale('log')
        plain_log(ax, 'x', list(x))
        # a range under a factor of 4 gets finer ticks, so that more than one is labeled
        cand = (0.5, 1, 2, 5, 10) if y.max() / y.min() > 4 else (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2, 2.5, 3)
        lo, hi = ax.get_ylim()
        plain_log(ax, 'y', [t for t in cand if lo <= t <= hi])
    ax.set_xlabel(xlabel); ax.set_ylabel(r'$\varepsilon^\infty/(\eta\sigma^2)$')
    h = [Line2D([], [], color=A['color'], ls=A['ls'], label='CNN, analytic'),
         Line2D([], [], color=B['color'], ls=B['ls'], label='LCN, analytic'),
         Line2D([], [], color=A['color'], ls='none', marker=A['marker'], label='CNN, ODE'),
         Line2D([], [], color=B['color'], ls='none', marker=B['marker'], label='LCN, ODE')]
    ax.legend(handles=h, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2,
              fontsize=12, handlelength=2.0, columnspacing=1.0)
    common.save(fig, name)


def plot(d):
    """Draw the three panels from the output of compute (or its cached npz)."""
    set_style()
    ca = np.linspace(0, 0.99, 200)
    panel(d['cs'], d['a'], ca, cf.floor_cnn(P_A, ca, T), np.full_like(ca, cf.floor_lcn(P_A, T)),
          r'Correlation $c$', 'noise-floor-a', False)
    Pa = np.linspace(3, 8, 100)
    for key, c in (('b', 0.0), ('c', 0.5)):
        panel(d['Ps'], d[key], Pa, cf.floor_cnn(Pa, c, T), cf.floor_lcn(Pa, T),
              r'Positions $P$', f'noise-floor-{key}', True)


def main(args):
    """Compute (unless args.replot) and cache the results, then draw the figure."""
    path = common.data_path('noise_floor', args.quick)
    if not args.replot:
        np.savez(path, **compute(args))
    plot(np.load(path))


if __name__ == '__main__':
    main(common.parse(__doc__))
