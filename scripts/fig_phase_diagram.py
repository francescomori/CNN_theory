"""Figure 6 (phase-diagram.pdf), Figure 8 (phase-eta.pdf, phase-eps.pdf) and Figure 9
(phase-var-a/b/c/d.pdf), Section 4.3 and Appendix I.

For each student, correlation c and training time alpha_F, eps*(alpha_F) is the lowest
eps_g(alpha_F) over constant learning rates eta. For every (student, c, eta) the ODEs are
integrated once from the D -> infinity random initial condition (R = 0, Q = I / k_s), and
eps_g is interpolated from the dense ODE output onto a log grid of alpha_F. At each alpha_F the best eta of the grid is
refined by a parabola through it and its two neighbors in (log eta, log eps).
Figures 6 and 8: g = erf, P = 4, k_s = k_t = 1, T = 1, sigma = 1, c = 0, 0.025, ..., 0.975,
80 values of eta from 1e-4 to 3, 400 values of alpha_F from 1e-2 to 1e4.
Figure 9: one parameter changed, (a) P = 8 (eta from 1e-5), (b) k_s = k_t = 2, (c) T = 2,
(d) sigma = 2.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import common                # first: puts the repository root on sys.path for cnnlcn
from cnnlcn import model, solve, closed_forms as cf
from cnnlcn.style import set_style, STUDENT, row_major

PAPER = (4, 1, 1.0, 1.0, 1e-4)                      # (P, k, T, sigma, smallest eta) of Figures 6 and 8
VARIANTS = {'a': (8, 1, 1.0, 1.0, 1e-5), 'b': (4, 2, 1.0, 1.0, 1e-4),
            'c': (4, 1, 2.0, 1.0, 1e-4), 'd': (4, 1, 1.0, 2.0, 1e-4)}
ALPHA_MAX = 1e4
VMAX = 1.25                                         # common color scale of the Figure 9 panels
C_APP = {0.0: ('-', 'o'), 0.5: ('--', 's'), 0.9: (':', '^')}    # Figure 8: c -> (line, marker)


def grids(quick, eta_min):
    """Grids of c, eta and alpha_F (coarser if quick). Returns three arrays.

    Inputs : quick (bool), eta_min (smallest eta; the eta grid keeps the log spacing of
             80 points in [1e-4, 3], or of 40 points if quick).
    """
    n_eta = int(round(79 * np.log(3.0 / eta_min) / np.log(3.0 / 1e-4))) + 1
    if quick:
        return (np.round(np.arange(0.0, 0.951, 0.1), 3), np.logspace(np.log10(eta_min), np.log10(3.0), n_eta // 2),
                np.logspace(-2, 3, 150))
    return (np.round(np.arange(0.0, 0.976, 0.025), 3), np.logspace(np.log10(eta_min), np.log10(3.0), n_eta),
            np.logspace(-2, np.log10(ALPHA_MAX), 400))


def run(job):
    """eps_g on the alpha_F grid for job = (student, c, eta, grid, P, k, T, sigma).

    Returns an array (len(grid),), nan where the integration failed or diverged.
    """
    st, c, eta, grid, P, k, T, sigma = job
    a = model.Arch(P, k, k, c, T)
    try:
        r = solve.integrate(a, st, model.init_large_D(a, st), eta, sigma, grid[-1], n_out=600)
    except Exception:
        return np.full(grid.size, np.nan)
    ok = np.isfinite(r['eps'])
    n = ok.size if ok.all() else np.argmin(ok)            # a diverging run is cut where it blows up
    if n < 2:
        return np.full(grid.size, np.nan)
    al, ep = r['alpha'][1:n], r['eps'][1:n]                # alpha[0] = 0 is not on the log grid
    return np.interp(np.log(grid), np.log(al), ep, left=np.nan, right=np.nan)


def best_over_eta(eps, etas):
    """Lowest error over eta at each alpha_F, refined by a parabola in (log eta, log eps).

    Inputs : eps (n_eta, n_alpha), nan for invalid runs; etas (n_eta,), log spaced.
    Outputs: eps* (n_alpha,) and eta* (n_alpha,).
    """
    x = np.log(etas)
    h = x[1] - x[0]
    ly = np.log(np.where(np.isfinite(eps), eps, np.inf))
    best, etab = np.empty(eps.shape[1]), np.empty(eps.shape[1])
    for j, i in enumerate(np.argmin(ly, 0)):
        best[j], etab[j] = np.exp(ly[i, j]), etas[i]
        if 0 < i < len(etas) - 1 and np.all(np.isfinite(ly[i - 1:i + 2, j])):
            ym, y0, yp = ly[i - 1:i + 2, j]
            curv = yp - 2 * y0 + ym
            if curv > 0:
                best[j] = np.exp(y0 - (yp - ym) ** 2 / (8 * curv))
                etab[j] = np.exp(x[i] - 0.5 * h * (yp - ym) / curv)
    return best, etab


def sweep(P, k, T, sigma, eta_min, args):
    """eps* and eta* of both students on the (c, alpha_F) grid.

    Inputs : P, k (= k_s = k_t), T, sigma, eta_min (smallest eta), args (quick, workers).
    Outputs: dict with cs, grid, best and eta_best (2, n_c, n_alpha), index 0 = CNN, 1 = LCN.
    """
    cs, etas, grid = grids(args.quick, eta_min)
    jobs = [(st, c, eta, grid, P, k, T, sigma) for st in ('cnn', 'lcn') for c in cs for eta in etas]
    eps = np.array(common.pmap(run, jobs, args.workers)).reshape(2, len(cs), len(etas), grid.size)
    opt = np.array([[best_over_eta(e, etas) for e in eps_st] for eps_st in eps])   # (2, n_c, 2, n_alpha)
    return dict(cs=cs, grid=grid, best=opt[:, :, 0], eta_best=opt[:, :, 1])


def label_position(lr, cs, grid, sign, candidates):
    """First candidate position (axes units) whose surroundings lie well inside the region sign * lr > 0.

    Inputs : lr (n_c, n_alpha) log ratio, cs, grid, sign (+1: LCN better, -1: CNN better),
             candidates (list of (x, y) in axes units). Points around the label must have
             sign * lr > 0.05.
    Outputs: (x, y), or None if no candidate fits.
    """
    lx0, lx1 = np.log10(grid[0]), np.log10(grid[-1])
    for x, y in candidates:
        pts = [(10 ** (lx0 + (x + dx) * (lx1 - lx0)), (y + dy) * cs[-1])
               for dx in (-0.05, 0.06, 0.17) for dy in (-0.09, 0.0, 0.09)]
        idx = [(min(np.searchsorted(cs, yy), len(cs) - 1), min(np.searchsorted(grid, xx), len(grid) - 1))
               for xx, yy in pts]
        if all(np.isfinite(lr[i, j]) and sign * lr[i, j] > 0.05 for i, j in idx):
            return x, y
    return None


def draw_map(ax, d, P, k, T, sigma, vmax, fontsize):
    """Draw log10(eps*_CNN / eps*_LCN), the exact boundary and the early-time line on ax.

    Inputs : ax, d (output of sweep), P, k, T, sigma (for c0), vmax (color range),
             fontsize (of the region labels). Outputs: the pcolormesh and the log ratio.
    """
    cs, grid = d['cs'], d['grid']
    lr = np.log10(d['best'][0] / d['best'][1])
    ce = np.r_[0.0, cs[1:] - (cs[1] - cs[0]) / 2, cs[-1] + (cs[1] - cs[0]) / 2]
    ge = np.r_[grid[0], np.sqrt(grid[1:] * grid[:-1]), grid[-1]]
    pc = ax.pcolormesh(ge, ce, lr, cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax), shading='flat', rasterized=True)
    ax.contour(grid, cs, lr, levels=[0], colors='k', linewidths=2.2)
    c0 = cf.c0(P, k, T, sigma)
    al = np.logspace(np.log10(grid[0]), np.log10(min(grid[-1], 1.0 / c0)), 200)
    ax.plot(al, c0 * al, color='C1', ls='--', lw=2.6)
    ax.set_xscale('log'); ax.set_xlim(grid[0], grid[-1]); ax.set_ylim(0, cs[-1])
    ax.set_xlabel(r'Training horizon $\alpha_F$'); ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    for sign, text, cand in ((+1, 'LCN\nbetter', [(0.03, 0.80), (0.03, 0.88), (0.15, 0.88)]),
                             (-1, 'CNN\nbetter', [(0.70, 0.45), (0.70, 0.25), (0.75, 0.70)])):
        pos = label_position(lr, cs, grid, sign, cand)
        if pos:
            ax.text(*pos, text, transform=ax.transAxes, fontsize=fontsize, va='center')
    return pc, lr


def plot_main(d):
    """Figure 6 from d, the output of sweep at the paper's parameters."""
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    lr = np.log10(d['best'][0] / d['best'][1])
    pc, _ = draw_map(ax, d, *PAPER[:4], vmax=np.nanmax(np.abs(lr)), fontsize=17)
    ax.set_ylabel(r'Correlation $c$')
    cb = fig.colorbar(pc, ax=ax, pad=0.03)
    cb.set_label(r'$\log_{10}(\varepsilon^*_{\rm CNN}/\varepsilon^*_{\rm LCN})$')
    h = [Line2D([], [], color='k', lw=2.2, label='Exact ODE'),
         Line2D([], [], color='C1', ls='--', lw=2.6, label='Early-time approximation')]
    ax.legend(handles=h, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=15)
    fig.tight_layout()
    common.save(fig, 'phase-diagram')


def plot_curves(d, key, ylabel, name):
    """Figure 8 panel `name`: d[key] (key 'eta_best' or 'best', y label ylabel) against alpha_F at c = 0, 0.5, 0.9."""
    cs, grid = d['cs'], d['grid']
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    every = max(1, grid.size // 9)
    for c, (ls, mk) in C_APP.items():
        i = int(np.argmin(np.abs(cs - c)))
        for j, st in enumerate(('cnn', 'lcn')):
            ax.plot(grid, d[key][j, i], color=STUDENT[st]['color'], ls=ls, marker=mk, ms=7,
                    markevery=(every // 2 * j, every))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel(r'Training horizon $\alpha_F$'); ax.set_ylabel(ylabel)
    h = [Line2D([], [], color=STUDENT[st]['color'], lw=2.2, label=st.upper()) for st in ('cnn', 'lcn')]
    h += [Line2D([], [], alpha=0, label=' ')]
    h += [Line2D([], [], color='0.3', ls=ls, marker=mk, ms=7, label=f'$c = {c:g}$') for c, (ls, mk) in C_APP.items()]
    ax.legend(handles=row_major(h, 3), frameon=False, loc='upper center', bbox_to_anchor=(0.42, -0.20), ncol=3,
              fontsize=14, handlelength=2.2, columnspacing=0.9)
    fig.tight_layout()
    common.save(fig, name)


def plot_variant(d, tag):
    """Figure 9 panel `tag` ('a' to 'd') from d, the output of sweep. All panels share size,
    axes positions and color scale."""
    P, k, T, sigma, _ = VARIANTS[tag]
    fig = plt.figure(figsize=(6.4, 4.9))
    ax = fig.add_axes([0.155, 0.175, 0.60, 0.795])
    pc, lr = draw_map(ax, d, P, k, T, sigma, vmax=VMAX, fontsize=16)
    ax.set_xticks([1e-2, 1, 1e2, 1e4])
    if tag in 'ac':                                          # y label on the left column only
        ax.set_ylabel(r'Correlation $c$')
    extend = 'min' if np.nanmin(lr) < -VMAX else 'neither'
    cb = fig.colorbar(pc, cax=fig.add_axes([0.785, 0.175, 0.035, 0.795]), extend=extend)
    cb.set_label(r'$\log_{10}(\varepsilon^*_{\rm CNN}/\varepsilon^*_{\rm LCN})$')
    common.save(fig, f'phase-var-{tag}')


def main(args):
    """Compute (unless args.replot) and cache the sweeps, then draw Figures 6, 8 and 9."""
    sets = {'phase_diagram': PAPER, **{f'phase_variant_{t}': v for t, v in VARIANTS.items()}}
    data = {}
    for name, (P, k, T, sigma, eta_min) in sets.items():
        path = common.data_path(name, args.quick)
        if not args.replot:
            np.savez(path, **sweep(P, k, T, sigma, eta_min, args))
        data[name] = np.load(path)
    set_style()
    plot_main(data['phase_diagram'])
    plot_curves(data['phase_diagram'], 'eta_best', r'Learning rate $\eta^*$', 'phase-eta')
    plot_curves(data['phase_diagram'], 'best', r'Generalization error $\varepsilon^*$', 'phase-eps')
    for t in VARIANTS:
        plot_variant(data[f'phase_variant_{t}'], t)


if __name__ == '__main__':
    main(common.parse(__doc__))
