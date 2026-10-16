"""Figure 2 (rescaling-a.pdf, rescaling-b.pdf): eps_g of the CNN at alpha and of the LCN at P alpha.

Lines: ODEs. Markers: one online SGD run per student, from the same initial weights as
the ODE. Insets: both ODE curves against their own alpha. (a) c = 0, (b) c = 0.5.
Parameters: g = erf, P = 4, k_t = 2, k_s = 3, eta = 1, sigma = 0, T = 1, SGD at D = 1000.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common                # first: puts the repository root on sys.path for cnnlcn
from cnnlcn import model, sgd, solve
from cnnlcn.style import set_style, STUDENT, FIGSIZE, MARGINS, plain_log

P, KT, KS, T, ETA, SIGMA, D = 4, 2, 3, 1.0, 1.0, 0.0, 1000
XMAX, EVERY = 24.0, 2.0          # CNN time range and SGD recording interval


def run(c):
    """SGD and ODE curves of both students at correlation c (LCN run P times longer).

    Inputs : c. Outputs: dict with {cnn, lcn}_{sgd, ode}_{a, e}: alpha and eps_g arrays.
    """
    a = model.Arch(P, KT, KS, c, T)
    out = {}
    for st, amax, every, seed in (('cnn', XMAX, EVERY, 10), ('lcn', P * XMAX, P * EVERY, 20)):
        W0 = sgd.init_weights(a, st, D, 1, np.random.default_rng(seed))
        R, Q = sgd.order_params(a, st, W0, D)
        s = sgd.simulate(a, st, D, ETA, SIGMA, amax, record_every=every, seed=seed + 1, W0=W0)
        o = solve.integrate(a, st, model.pack(R[0], Q[0]), ETA, SIGMA, P * XMAX, n_out=600, log_grid=False)
        out.update({f'{st}_sgd_a': s['alpha'], f'{st}_sgd_e': s['eps'][0],
                    f'{st}_ode_a': o['alpha'], f'{st}_ode_e': o['eps']})
    return out


def plot(d, tag):
    """Draw panel `tag` ('a' or 'b') from the output of run (or its cached npz)."""
    set_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINS)
    A, B = STUDENT['cnn'], STUDENT['lcn']
    ax.plot(d['cnn_ode_a'], d['cnn_ode_e'], color=A['color'], ls=A['ls'], lw=2.4)
    ax.plot(d['lcn_ode_a'] / P, d['lcn_ode_e'], color=B['color'], ls=B['ls'], lw=2.0)
    ax.plot(d['cnn_sgd_a'], d['cnn_sgd_e'], ls='none', marker=A['marker'], mfc='none', mec=A['color'], mew=1.6)
    ax.plot(d['lcn_sgd_a'] / P, d['lcn_sgd_e'], ls='none', marker=B['marker'], mfc='none', mec=B['color'], mew=1.6)
    ax.set_yscale('log'); ax.set_xlim(0, XMAX)
    # raise the top of the log axis so that no curve or marker runs under the inset
    lo = min(d['cnn_ode_e'].min(), d['lcn_ode_e'].min())
    hi = max(d['cnn_ode_e'].max(), d['lcn_ode_e'].max())
    curves = [(d[f'{st}_{k}_a'] / (P if st == 'lcn' else 1), d[f'{st}_{k}_e'])
              for st in ('cnn', 'lcn') for k in ('ode', 'sgd')]
    right = max(e[a >= 0.45 * XMAX].max() for a, e in curves)
    bottom = 0.85 * min(lo, d['cnn_sgd_e'].min(), d['lcn_sgd_e'].min())
    ax.set_ylim(bottom, max(1.15 * max(hi, d['cnn_sgd_e'].max(), d['lcn_sgd_e'].max()),
                            bottom * (right / bottom) ** (1 / 0.42)))
    plain_log(ax, 'y', [t for t in (0.2, 0.3, 0.5, 1, 2, 3, 5, 10) if 0.8 * lo <= t <= 1.2 * hi])
    ax.set_xlabel(r'Training time $\alpha$'); ax.set_ylabel(r'Generalization error $\varepsilon_g$')
    ins = ax.inset_axes([0.47, 0.57, 0.49, 0.39])
    ins.plot(d['cnn_ode_a'], d['cnn_ode_e'], color=A['color'], lw=1.6)
    ins.plot(d['lcn_ode_a'], d['lcn_ode_e'], color=B['color'], ls=B['ls'], lw=1.6)
    ins.set_xlim(0, P * XMAX); ins.set_xticks([0, P * XMAX / 2, P * XMAX])
    ins.tick_params(labelsize=11); ins.set_xlabel(r'$\alpha$', fontsize=12, labelpad=1)
    eA, eB = r'$\varepsilon_g^{\rm CNN}(\alpha)$', r'$\varepsilon_g^{\rm LCN}(P\alpha)$'
    h = [Line2D([], [], color=A['color'], ls=A['ls'], lw=2.4, label=f'{eA} (ODE)'),
         Line2D([], [], color=B['color'], ls=B['ls'], lw=2.0, label=f'{eB} (ODE)'),
         Line2D([], [], ls='none', marker=A['marker'], mfc='none', mec=A['color'], mew=1.6, label=f'{eA} (SGD)'),
         Line2D([], [], ls='none', marker=B['marker'], mfc='none', mec=B['color'], mew=1.6, label=f'{eB} (SGD)')]
    ax.legend(handles=h, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2,
              fontsize=12, handlelength=2.0, columnspacing=1.0)
    common.save(fig, f'rescaling-{tag}')


def main(args):
    """Compute (unless args.replot) and cache the results, then draw both panels."""
    for tag, c in (('a', 0.0), ('b', 0.5)):
        path = common.data_path(f'rescaling_{tag}', args.quick)
        if not args.replot:
            np.savez(path, **run(c))
        plot(np.load(path), tag)


if __name__ == '__main__':
    main(common.parse(__doc__))
