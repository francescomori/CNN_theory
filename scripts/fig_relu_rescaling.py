"""Figure 11, Appendix J (relu-rescaling-a.pdf, relu-rescaling-b.pdf): Figure 2 with the ReLU
activation. eps_g of the CNN at alpha and of the LCN at P alpha, from online SGD averaged over
16 runs per student (student weights iid N(0, 1)). Insets: both curves against their own
alpha. (a) c = 0, (b) c = 0.5.
Parameters: g(z) = max(0, z), P = 4, k_t = 2, k_s = 3, eta = 0.2, sigma = 0, T = 1, D = 1000.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import common                # first: puts the repository root on sys.path for cnnlcn
from cnnlcn import model, sgd
from cnnlcn.style import set_style, STUDENT, FIGSIZE, MARGINS, plain_log

P, KT, KS, T, ETA, SIGMA, D = 4, 2, 3, 1.0, 0.2, 0.0, 1000
XMAX = 60.0
N_RUNS = 16


def run(c, n_runs):
    """SGD of both students at correlation c, up to alpha = P XMAX.

    Inputs : c, n_runs. Outputs: dict with {cnn, lcn}_a (n,) and {cnn, lcn}_e (n_runs, n).
    """
    a = model.Arch(P, KT, KS, c, T)
    out = {}
    for st, every, seed in (('cnn', 1.0, 10), ('lcn', P * 1.0, 20)):   # LCN record k is at alpha = P k
        r = sgd.simulate(a, st, D, ETA, SIGMA, P * XMAX, record_every=every, n_chains=n_runs,
                         seed=seed, act='relu')
        out[f'{st}_a'], out[f'{st}_e'] = r['alpha'], r['eps']
    return out


def plot(d, tag):
    """Draw panel `tag` ('a' or 'b') from the output of run (or its cached npz)."""
    set_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINS)
    A, B = STUDENT['cnn'], STUDENT['lcn']
    mc, ml = d['cnn_e'].mean(0), d['lcn_e'].mean(0)
    m = d['cnn_a'] <= XMAX
    ax.plot(d['cnn_a'][m], mc[m], **A, markevery=6, lw=2.2)
    ax.plot(d['lcn_a'] / P, ml, **B, markevery=(3, 6), mfc='none', lw=2.0)
    ax.set_yscale('log'); ax.set_xlim(0, XMAX)
    lo, hi = min(mc[m].min(), ml.min()), max(mc[m].max(), ml.max())
    plain_log(ax, 'y', [t for t in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5) if 0.8 * lo <= t <= 1.2 * hi])
    ax.set_xlabel(r'Training time $\alpha$'); ax.set_ylabel(r'Generalization error $\varepsilon_g$')
    ins = ax.inset_axes([0.47, 0.57, 0.49, 0.39])
    ins.plot(d['cnn_a'], mc, color=A['color'], lw=1.6)
    ins.plot(d['lcn_a'], ml, color=B['color'], ls=B['ls'], lw=1.6)
    ins.set_xlim(0, P * XMAX); ins.set_xticks([0, P * XMAX / 2, P * XMAX])
    ins.tick_params(labelsize=11); ins.set_xlabel(r'$\alpha$', fontsize=12, labelpad=1)
    h = [Line2D([], [], **A, lw=2.2, label=r'$\varepsilon_g^{\rm CNN}(\alpha)$ (SGD)'),
         Line2D([], [], **B, mfc='none', lw=2.0, label=r'$\varepsilon_g^{\rm LCN}(P\alpha)$ (SGD)')]
    ax.legend(handles=h, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2,
              fontsize=12, handlelength=2.2, columnspacing=1.0)
    common.save(fig, f'relu-rescaling-{tag}')


def main(args):
    """Compute (unless args.replot) and cache the results, then draw both panels."""
    for tag, c in (('a', 0.0), ('b', 0.5)):
        path = common.data_path(f'relu_rescaling_{tag}', args.quick)
        if not args.replot:
            np.savez(path, **run(c, 4 if args.quick else N_RUNS))
        plot(np.load(path), tag)


if __name__ == '__main__':
    main(common.parse(__doc__))
