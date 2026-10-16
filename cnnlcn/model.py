"""Order parameters, generalization error and ODEs of the CNN and LCN students.

Units and conventions are those of the paper (Secs. 2-3, Apps. A, B, G):

    data      x = (x_0, ..., x_{P-1}), x_p in R^D, E[x_{p,a} x_{q,b}] = S_pq delta_ab,
              S_pp = 1, S_pq = c for p != q. Positions are taken mod P.
    teacher   lambda_p = (k_t D)^{-1/2} sum_{l < k_t} w_l . x_{p+l},  w_l . w_l' / D = T delta_ll'
    students  hat lambda_p = (k_s D)^{-1/2} sum_{l < k_s} hat w_l . x_{p+l}
              (CNN: one kernel, LCN: one kernel hat w^{(p)} per position),
              outputs sum_p g(lambda_p) and sum_p g(hat lambda_p), g(z) = erf(z / sqrt 2)
    error     eps_g = E[(f* - hat y)^2]

Order parameters, flattened into one state vector y = (R, Q):
    CNN  R[l, l']       = hat w_l . w_l' / (sqrt(k_s k_t) D)          shape (k_s, k_t)
         Q[l, l']       = hat w_l . hat w_l' / (k_s D)                shape (k_s, k_s)
    LCN  R[p, l, l']    = hat w^{(p)}_l . w_l' / (sqrt(k_s k_t) D)    shape (P, k_s, k_t)
         Q[p, q, l, l'] = hat w^{(p)}_l . hat w^{(q)}_l' / (k_s D)    shape (P, P, k_s, k_s)

The ODEs read dy/dalpha = (eta / k_s) F(y) + (eta^2 / k_s^2) G(y; sigma), with alpha = t / D.
"""
import numpy as np
from .integrals import I2, J2, I3, I4, I2_relu


class Arch:
    """Model parameters and the index tables used by the ODEs.

    Inputs : P (positions), kt (teacher kernel size), ks (student kernel size),
             c (correlation between positions, 0 <= c < 1), T (teacher norm).
    Attributes: P, kt, ks, c, T; S (P, P) position covariance; Tt (kt, kt) teacher
             overlaps w_l . w_l' / (k_t D); rs (P, ks), rt (P, kt): input position read by
             position p at tap l; S between these positions (Sss, Sst, Ssr, Str); Ctt (P, P)
             covariance of the teacher preactivations.
    """

    def __init__(self, P, kt, ks, c=0.0, T=1.0):
        if not 1 <= kt <= ks <= P or not 0.0 <= c < 1.0:
            raise ValueError(f'need 1 <= kt <= ks <= P and 0 <= c < 1, got {P, kt, ks, c}')
        self.P, self.kt, self.ks, self.c, self.T = int(P), int(kt), int(ks), float(c), float(T)
        S = (1.0 - c) * np.eye(P) + c * np.ones((P, P))
        self.S = S
        self.Tt = (T / kt) * np.eye(kt)
        self.rs = (np.arange(P)[:, None] + np.arange(ks)[None, :]) % P
        self.rt = (np.arange(P)[:, None] + np.arange(kt)[None, :]) % P
        rs, rt = self.rs, self.rt
        self.Sss = S[rs[:, :, None, None], rs[None, None, :, :]]   # (P, ks, P, ks)
        self.Sst = S[rs[:, :, None, None], rt[None, None, :, :]]   # (P, ks, P, kt)
        self.Ssr = S[rs]                                            # (P, ks, P)
        self.Str = S[rt]                                            # (P, kt, P)
        Stt = S[rt[:, :, None, None], rt[None, None, :, :]]        # (P, kt, P, kt)
        self.Ctt = np.einsum('lm,plqm->pq', self.Tt, Stt)


def pack(R, Q):
    """Flat state vector from the order parameters R, Q (arrays). Returns a 1-d array."""
    return np.concatenate([np.ravel(R), np.ravel(Q)])


def unpack(arch, student, y):
    """Order parameters from a flat state.

    Inputs : arch (Arch), student ('cnn' or 'lcn'), y (flat state).
    Outputs: R, Q with the shapes given in the module docstring.
    """
    P, kt, ks = arch.P, arch.kt, arch.ks
    if student == 'cnn':
        return y[:ks * kt].reshape(ks, kt), y[ks * kt:].reshape(ks, ks)
    n = P * ks * kt
    return y[:n].reshape(P, ks, kt), y[n:].reshape(P, P, ks, ks)


def field_cov(arch, student, R, Q):
    """Covariances of the preactivations.

    Inputs : arch, student, R, Q.
    Outputs: Css[p, q] = <hat lambda_p hat lambda_q> and Cst[p, q] = <hat lambda_p lambda_q>,
             both (P, P). The teacher block <lambda_p lambda_q> is arch.Ctt.
    """
    if student == 'cnn':
        return (np.einsum('lm,plqm->pq', Q, arch.Sss), np.einsum('lm,plqm->pq', R, arch.Sst))
    return (np.einsum('pqlm,plqm->pq', Q, arch.Sss), np.einsum('plm,plqm->pq', R, arch.Sst))


def gen_error_cov(Ctt, Cst, Css, act='erf'):
    """Generalization error from the preactivation covariances (App. G).

    Inputs : Ctt, Cst, Css (P, P), see field_cov; act, the activation ('erf' or 'relu').
    Outputs: eps_g (float).
    """
    K = I2 if act == 'erf' else I2_relu
    dt, ds = np.diag(Ctt), np.diag(Css)
    return float(K(dt[:, None], Ctt, dt[None, :]).sum()
                 - 2.0 * K(ds[:, None], Cst, dt[None, :]).sum()
                 + K(ds[:, None], Css, ds[None, :]).sum())


def gen_error(arch, student, y):
    """Generalization error eps_g (float) of the flat state y of a student ('cnn' or 'lcn')."""
    Css, Cst = field_cov(arch, student, *unpack(arch, student, y))
    return gen_error_cov(arch.Ctt, Cst, Css)


def eta2_average(Ctt, Cst, Css, sigma):
    """The average V[p, q] = <delta^2 g'(hat lambda_p) g'(hat lambda_q)> in the O(eta^2) term.

    With delta = sum_r g(lambda_r) + sigma xi - sum_r g(hat lambda_r), it is
        sum_{r,s} [I4(hp, hq, tr, ts) - 2 I4(hp, hq, tr, hs) + I4(hp, hq, hr, hs)] + sigma^2 J2(hp, hq),
    where h stands for a student and t for a teacher preactivation. Shared by both students.
    Inputs : Ctt, Cst, Css (P, P), sigma (label-noise standard deviation).
    Outputs: V (P, P).
    """
    dt, ds = np.diag(Ctt), np.diag(Css)
    c11, c22, c12 = ds[:, None, None, None], ds[None, :, None, None], Css[:, :, None, None]
    # covariances of hat lambda_p, hat lambda_q with the field in slot 3 (index r) or 4 (index s)
    h3 = dict(c13=Css[:, None, :, None], c23=Css[None, :, :, None], c33=ds[None, None, :, None])
    t3 = dict(c13=Cst[:, None, :, None], c23=Cst[None, :, :, None], c33=dt[None, None, :, None])
    h4 = dict(c14=Css[:, None, None, :], c24=Css[None, :, None, :], c44=ds[None, None, None, :])
    t4 = dict(c14=Cst[:, None, None, :], c24=Cst[None, :, None, :], c44=dt[None, None, None, :])

    def summed_I4(f3, f4, c34):
        """sum_{r,s} I4(hat lambda_p, hat lambda_q, f3_r, f4_s), as a (P, P) array."""
        return I4(c11, c12, f3['c13'], f4['c14'], c22, f3['c23'], f4['c24'],
                  f3['c33'], c34, f4['c44']).sum(axis=(2, 3))

    tt = summed_I4(t3, t4, Ctt[None, None, :, :])
    th = summed_I4(t3, h4, Cst.T[None, None, :, :])      # <lambda_r hat lambda_s> = Cst[s, r]
    hh = summed_I4(h3, h4, Css[None, None, :, :])
    return tt - 2.0 * th + hh + sigma ** 2 * J2(ds[:, None], Css, ds[None, :])


def rhs_cnn(arch, y, eta, sigma, eta2=True):
    """dy/dalpha for the CNN (Result 3.1, App. A).

    Inputs : arch, y (flat CNN state), eta (learning rate), sigma (label noise),
             eta2 (include the O(eta^2) term).
    Outputs: flat array, same layout as y.
    """
    P, kt, ks = arch.P, arch.kt, arch.ks
    R, Q = unpack(arch, 'cnn', y)
    Css, Cst = field_cov(arch, 'cnn', R, Q)
    ds, dt = np.diag(Css), np.diag(arch.Ctt)
    # covariances of a preactivation with the projection of the input at position r on a
    # teacher weight (mu) or a student weight (nu)
    Hmu = np.einsum('nm,pnr->prm', R, arch.Ssr)          # <hat lambda_p mu^(m)_r>  (P, P, kt)
    Tmu = np.einsum('nm,qnr->qrm', arch.Tt, arch.Str)    # <lambda_q mu^(m)_r>      (P, P, kt)
    Hnu = np.einsum('nm,pnr->prm', Q, arch.Ssr)          # <hat lambda_p nu^(m)_r>  (P, P, ks)
    Tnu = np.einsum('mn,qnr->qrm', R, arch.Str)          # <lambda_q nu^(m)_r>      (P, P, ks)
    # axes (p, q, l, m), where r = rs[p, l] is the input position read by position p at tap l
    p, q = np.arange(P)[:, None, None, None], np.arange(P)[None, :, None, None]
    r = arch.rs[:, None, :, None]

    def drive(Hx, Tx, n):
        """sum_{p,q} [I3(hat lambda_p, x_r, lambda_q) - I3(hat lambda_p, x_r, hat lambda_q)], shape (ks, n).

        x_r is the projection mu_r (onto a teacher weight, n = kt) or nu_r (onto a student
        weight, n = ks); Hx, Tx are its covariances with hat lambda and lambda (Hmu, Tmu or Hnu, Tnu).
        """
        m = np.arange(n)[None, None, None, :]
        c11, c12 = ds[p], Hx[p, r, m]
        return (I3(c11, c12, Cst[p, q], Tx[q, r, m], dt[q])
                - I3(c11, c12, Css[p, q], Hx[q, r, m], ds[q])).sum(axis=(0, 1))

    dR = (eta / ks) * drive(Hmu, Tmu, kt)
    A = drive(Hnu, Tnu, ks)
    dQ = (eta / ks) * (A + A.T)
    if eta2:
        V = eta2_average(arch.Ctt, Cst, Css, sigma)
        dQ = dQ + (eta ** 2 / ks ** 2) * np.einsum('plqm,pq->lm', arch.Sss, V)
    return pack(dR, dQ)


def rhs_lcn(arch, y, eta, sigma, eta2=True):
    """dy/dalpha for the LCN (Result 3.1, App. B).

    Inputs : arch, y (flat LCN state), eta, sigma, eta2 (include the O(eta^2) term).
    Outputs: flat array, same layout as y.
    """
    P, kt, ks = arch.P, arch.kt, arch.ks
    R, Q = unpack(arch, 'lcn', y)
    Css, Cst = field_cov(arch, 'lcn', R, Q)
    ds, dt = np.diag(Css), np.diag(arch.Ctt)
    Hmu = np.einsum('pnm,pnr->prm', R, arch.Ssr)         # <hat lambda_p mu^(m)_r>      (P, P, kt)
    Tmu = np.einsum('nm,qnr->qrm', arch.Tt, arch.Str)    # <lambda_q mu^(m)_r>          (P, P, kt)
    Hnu = np.einsum('pqnm,pnr->pqrm', Q, arch.Ssr)       # <hat lambda_p nu^(q,m)_r>    (P, P, P, ks)
    Tnu = np.einsum('qmn,snr->sqrm', R, arch.Str)        # <lambda_s nu^(q,m)_r>        (P, P, P, ks)

    # dR^(p): axes (p, s, l, m), summed over s only
    p, s = np.arange(P)[:, None, None, None], np.arange(P)[None, :, None, None]
    r, m = arch.rs[:, None, :, None], np.arange(kt)[None, None, None, :]
    c11, c12 = ds[p], Hmu[p, r, m]
    dR = (eta / ks) * (I3(c11, c12, Cst[p, s], Tmu[s, r, m], dt[s])
                       - I3(c11, c12, Css[p, s], Hmu[s, r, m], ds[s])).sum(axis=1)

    # dQ^(p,q): axes (p, q, l, m, s), summed over s
    p, q = np.arange(P)[:, None, None, None, None], np.arange(P)[None, :, None, None, None]
    r, m = arch.rs[:, None, :, None, None], np.arange(ks)[None, None, None, :, None]
    s = np.arange(P)[None, None, None, None, :]
    c11, c12 = ds[p], Hnu[p, q, r, m]
    A = (I3(c11, c12, Cst[p, s], Tnu[s, q, r, m], dt[s])
         - I3(c11, c12, Css[p, s], Hnu[s, q, r, m], ds[s])).sum(axis=4)
    dQ = (eta / ks) * (A + A.transpose(1, 0, 3, 2))
    if eta2:
        V = eta2_average(arch.Ctt, Cst, Css, sigma)
        dQ = dQ + (eta ** 2 / ks ** 2) * np.einsum('plqm,pq->pqlm', arch.Sss, V)
    return pack(dR, dQ)


def rhs(arch, student, y, eta, sigma, eta2=True):
    """dy/dalpha of a student ('cnn' or 'lcn'); arguments and output as in rhs_cnn / rhs_lcn."""
    return (rhs_cnn if student == 'cnn' else rhs_lcn)(arch, y, eta, sigma, eta2)


def lcn_from_cnn(arch, R, Q):
    """Flat LCN state with the CNN kernel of order parameters R, Q copied to every position."""
    P, kt, ks = arch.P, arch.kt, arch.ks
    return pack(np.broadcast_to(R, (P, ks, kt)), np.broadcast_to(Q, (P, P, ks, ks)))


def perfect_recovery(arch, student):
    """Zero-error state hat w_l = sqrt(k_s / k_t) w_l (same kernel at every LCN position).

    Inputs : arch, student ('cnn' or 'lcn').
    Outputs: flat state, with R = Q = T / k_t on the first k_t taps and 0 elsewhere.
    """
    kt, ks, T = arch.kt, arch.ks, arch.T
    R, Q = np.zeros((ks, kt)), np.zeros((ks, ks))
    R[range(kt), range(kt)] = Q[range(kt), range(kt)] = T / kt
    return pack(R, Q) if student == 'cnn' else lcn_from_cnn(arch, R, Q)


def alignment(arch, R, Q):
    """Alignment gamma_s of the CNN kernel with each zero-error solution s (Sec. 3.2).

    gamma_s = sum_{j < k_t} R[s + j, j] / sqrt(T tr Q) is the cosine between the student
    kernel and the teacher kernel shifted by s taps.
    Inputs : arch, R (ks, kt), Q (ks, ks).
    Outputs: array (ks - kt + 1,), gamma_s for s = 0, ..., ks - kt.
    """
    kt, ks = arch.kt, arch.ks
    return np.array([np.trace(R[s:s + kt]) for s in range(ks - kt + 1)]) / np.sqrt(arch.T * np.trace(Q))


def random_init(arch, student, D, seed):
    """Order parameters of random student weights (iid N(0, 1) components) at dimension D.

    The result is an exact sample of the order parameters of such weights, for a teacher
    w_l = sqrt(T D) e_l: the components along the k_t teacher directions are drawn
    explicitly, the other D - k_t directions enter only through a Wishart matrix (Bartlett
    decomposition), so any D costs the same.
    Inputs : arch, student ('cnn' or 'lcn'; independent kernels at the LCN positions),
             D (feature dimension), seed (int).
    Outputs: flat state.
    """
    P, kt, ks, T = arch.P, arch.kt, arch.ks, arch.T
    n = ks if student == 'cnn' else P * ks                         # number of student weight vectors
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, kt))                                   # components along the teacher
    B = np.zeros((n, n))                                           # Bartlett factor of the rest
    for i in range(n):
        B[i, i] = np.sqrt(rng.chisquare(D - kt - i))
        B[i, :i] = rng.normal(size=i)
    Q = (Z @ Z.T + B @ B.T) / (ks * D)
    R = np.sqrt(T) * Z / (np.sqrt(ks * kt) * np.sqrt(D))
    if student == 'cnn':
        return pack(R, Q)
    return pack(R.reshape(P, ks, kt), Q.reshape(P, ks, P, ks).transpose(0, 2, 1, 3))


def init_large_D(arch, student):
    """Limit D -> infinity of random_init: R = 0, Q = I / k_s (LCN: Q^(p,q) = delta_pq I / k_s).

    Inputs : arch, student ('cnn' or 'lcn').
    Outputs: flat state.
    """
    P, kt, ks = arch.P, arch.kt, arch.ks
    if student == 'cnn':
        return pack(np.zeros((ks, kt)), np.eye(ks) / ks)
    return pack(np.zeros((P, ks, kt)), np.einsum('pq,lm->pqlm', np.eye(P), np.eye(ks) / ks))
