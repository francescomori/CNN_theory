"""Online SGD for the CNN and LCN students, batched over independent runs ("chains").

The model is that of the paper (Sec. 2), with teacher w_l = sqrt(T D) e_l and inputs
x = L z, L L^T = S, z iid N(0, 1). One SGD step on a fresh sample is
    hat w_l      <- hat w_l      + eta delta (k_s D)^{-1/2} sum_p g'(hat lambda_p) x_{p+l}   (CNN)
    hat w^(p)_l  <- hat w^(p)_l  + eta delta (k_s D)^{-1/2} g'(hat lambda_p) x_{p+l}         (LCN)
with delta = sum_p g(lambda_p) + sigma xi - sum_p g(hat lambda_p). The generalization
error is computed exactly from the weights, through their order parameters.
"""
import numpy as np
from scipy.special import erf
from . import model


def activation(name):
    """(g, g') for name 'erf' (g(z) = erf(z / sqrt 2)) or 'relu'. Returns two functions."""
    if name == 'erf':
        return (lambda z: erf(z / np.sqrt(2.0))), (lambda z: np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z * z))
    return (lambda z: np.maximum(z, 0.0)), (lambda z: (z > 0).astype(float))


def init_weights(arch, student, D, n_chains, rng, init='random'):
    """Initial student weights.

    Inputs : arch, student ('cnn' or 'lcn'), D, n_chains, rng (numpy Generator),
             init ('random': iid N(0, 1) components, independent across LCN positions;
             'perfect': the zero-error state hat w_l = sqrt(k_s / k_t) w_l).
    Outputs: array (n_chains, ks, D) for the CNN, (n_chains, P, ks, D) for the LCN.
    """
    P, kt, ks, T = arch.P, arch.kt, arch.ks, arch.T
    shape = (n_chains, ks, D) if student == 'cnn' else (n_chains, P, ks, D)
    if init == 'random':
        return rng.normal(size=shape)
    W = np.zeros(shape)
    W[..., range(kt), range(kt)] = np.sqrt(ks / kt) * np.sqrt(T * D)
    return W


def order_params(arch, student, W, D):
    """Order parameters R, Q of every chain (leading chain axis; shapes as in cnnlcn.model).

    Inputs : arch, student, W (weights, see init_weights), D.
    """
    kt, ks, T = arch.kt, arch.ks, arch.T
    f = np.sqrt(T / (ks * kt * D))           # the teacher is sqrt(T D) e_l, so R reads the first k_t components
    if student == 'cnn':
        return W[:, :, :kt] * f, np.einsum('bld,bmd->blm', W, W) / (ks * D)
    return W[:, :, :, :kt] * f, np.einsum('bpld,bqmd->bpqlm', W, W) / (ks * D)


def gen_errors(archs, student, W, D, act):
    """Exact generalization error of every chain.

    Inputs : archs (one Arch per chain), student, W (weights), D, act ('erf' or 'relu').
    Outputs: array (n_chains,).
    """
    R, Q = order_params(archs[0], student, W, D)
    out = np.empty(len(R))
    for b, a in enumerate(archs):
        Css, Cst = model.field_cov(a, student, R[b], Q[b])
        out[b] = model.gen_error_cov(a.Ctt, Cst, Css, act)
    return out


def simulate(arch, student, D, eta, sigma, alpha_max, record_every=1.0, n_chains=1, seed=0,
             init='random', act='erf', cs=None, W0=None):
    """Run online SGD and record eps_g every record_every units of alpha = t / D.

    Inputs : arch (sets P, kt, ks, T and c), student ('cnn' or 'lcn'), D, eta, sigma,
             alpha_max, record_every, n_chains, seed, init (see init_weights),
             act ('erf' or 'relu'), cs (optional: one correlation c per chain, overriding
             arch.c), W0 (optional initial weights, overriding init).
    Outputs: dict with alpha (n_rec,) and eps (n_chains, n_rec).
    """
    P, kt, ks, T = arch.P, arch.kt, arch.ks, arch.T
    rng = np.random.default_rng(seed)
    g, gp = activation(act)
    W = init_weights(arch, student, D, n_chains, rng, init) if W0 is None else np.array(W0, float)
    archs = [model.Arch(P, kt, ks, c, T) for c in ([arch.c] * n_chains if cs is None else cs)]
    L = np.stack([np.linalg.cholesky(a.S) for a in archs])          # (n_chains, P, P)
    a_s, a_t = np.sqrt(ks * D), np.sqrt(T / kt)
    every = max(1, int(round(record_every * D)))
    alphas, eps = [0.0], [gen_errors(archs, student, W, D, act)]
    for t in range(1, int(round(alpha_max * D)) + 1):
        x = np.einsum('bpq,bqd->bpd', L, rng.normal(size=(n_chains, P, D)))
        X = x[:, arch.rs, :]                                         # patches (n_chains, P, ks, D)
        lam = a_t * x[:, arch.rt, np.arange(kt)].sum(axis=2)          # teacher (n_chains, P)
        lam_h = np.einsum('bld,bpld->bp' if student == 'cnn' else 'bpld,bpld->bp', W, X) / a_s
        delta = g(lam).sum(1) + sigma * rng.normal(size=n_chains) - g(lam_h).sum(1)
        h = (eta / a_s) * delta[:, None] * gp(lam_h)                  # (n_chains, P)
        if student == 'cnn':
            W += np.einsum('bp,bpld->bld', h, X)
        else:
            W += h[:, :, None, None] * X
        if t % every == 0:
            alphas.append(t / D)
            eps.append(gen_errors(archs, student, W, D, act))
    return dict(alpha=np.array(alphas), eps=np.array(eps).T)
