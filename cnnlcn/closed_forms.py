"""Closed-form results for g(z) = erf(z / sqrt 2) and compound-symmetric S.

Early-time expansion (Result 4.1, App. F). From the D -> infinity random initial state
(R = 0, Q = I / k_s),
    eps_g(alpha) = eps_0 - (eta A - eta^2 B) alpha + O(alpha^2),   A, B > 0.
The names (H, V, K, n, chi, s_c, omega) follow App. F.
Noise floors (Result 4.2, App. H).
"""
import numpy as np

SQRT3 = np.sqrt(3.0)


def omega(chi):
    """omega(chi) = (4 - chi^2)^{-1/2}, for chi a float or array."""
    return 1.0 / np.sqrt(4.0 - chi ** 2)


def eps_T(P, c, T):
    """Teacher output power E[f*^2] for P positions, correlation c and teacher norm T (App. F)."""
    return (2.0 * P / np.pi) * (np.arcsin(T / (1.0 + T)) + (P - 1) * np.arcsin(T * c / (1.0 + T)))


def eps0(student, P, c, T):
    """Initial error eps_0 of student 'cnn' or 'lcn' at P, c, T (Result 4.1)."""
    e = eps_T(P, c, T) + P / 3.0
    return e + (2.0 * P * (P - 1) / np.pi) * np.arcsin(c / 2.0) if student == 'cnn' else e


def i4_patterns(chi):
    """The nine values of I4 at the initial state (table of App. F).

    Inputs : chi, correlation of distinct student preactivations (c for the CNN, 0 for the LCN).
    Outputs: dict pattern -> value. The letters name the preactivations in the slots
             (g', g', g, g) of I4, equal letters meaning the same preactivation.
    """
    h3, hx = 4.0 / (np.pi ** 2 * SQRT3), 4.0 * omega(chi) / np.pi ** 2
    return {
        'uuuu': h3 * np.arcsin(0.25),
        'uuuv': h3 * np.arcsin(chi / (2.0 * np.sqrt(2.0 * (3.0 - chi ** 2)))),
        'uuvv': h3 * np.arcsin((3.0 - 2.0 * chi ** 2) / (2.0 * (3.0 - chi ** 2))),
        'uuvw': h3 * np.arcsin(chi * (3.0 - 2.0 * chi) / (2.0 * (3.0 - chi ** 2))),
        'uvuu': hx * np.arcsin((2.0 - chi ** 2) / (2.0 * (3.0 - chi ** 2))),
        'uvuv': hx * np.arcsin(chi / (2.0 * (3.0 - chi ** 2))),
        'uvww': hx * np.arcsin((2.0 + chi - 2.0 * chi ** 2) / (2.0 * (2.0 - chi) * (1.0 + chi))),
        'uvuw': hx * np.arcsin(chi / (2.0 * np.sqrt((3.0 - chi ** 2) * (1.0 + chi)))),
        'uvwz': hx * np.arcsin(chi / (2.0 * (1.0 + chi))),
    }


def V(P, chi, c, T, sigma):
    """(V_d, V_o): V_pq = <delta^2 g'(hat lambda_p) g'(hat lambda_q)> at the initial state (App. F).

    Inputs : P, chi (correlation of student preactivations), c, T, sigma.
    Outputs: values for p = q and for p != q.
    """
    I = i4_patterns(chi)
    K_d = I['uuuu'] + 2 * (P - 1) * I['uuuv'] + (P - 1) * I['uuvv'] + (P - 1) * (P - 2) * I['uuvw']
    K_o = (2 * I['uvuu'] + 2 * I['uvuv'] + (P - 2) * I['uvww'] + 4 * (P - 2) * I['uvuw']
           + (P - 2) * (P - 3) * I['uvwz'])
    E = eps_T(P, c, T) + sigma ** 2
    return E * 2.0 / (np.pi * SQRT3) + K_d, E * (2.0 / np.pi) * omega(chi) + K_o


def early_coeffs(student, P, ks, c, T, sigma):
    """Coefficients (A, B) of the early-time expansion (Result 4.1, App. F).

    Inputs : student ('cnn' or 'lcn'), P, ks, c, T, sigma.
    Outputs: A, B (floats or arrays).
    """
    s_c = 1.0 + (P - 1) * c
    f = 1.0 - 1.0 / ks
    if student == 'cnn':
        w = omega(c)
        H_d = (P / np.pi) * (1.0 / SQRT3 + (P - 1) * c * w)
        H_o = (P / np.pi) * (c / SQRT3 + (2.0 + 2.0 * (P - 2) * c - (P - 1) * c ** 2) * w)
        V_d, V_o = V(P, c, c, T, sigma)
        A = 4.0 * T * P ** 2 * s_c ** 2 / (np.pi ** 2 * (1.0 + T)) + 2.0 * (H_d ** 2 / ks + f * H_o ** 2)
        B = P * (V_d * (H_d / ks + f * c * H_o) + V_o * ((P - 1) * c * H_d / ks + f * (1.0 + (P - 2) * c) * H_o))
        return A, B
    n_d = c ** 2 + (1.0 - c ** 2) / ks
    n_o = (1.0 - c ** 2) * f + (P - 1) * c ** 2
    V_d, V_o = V(P, 0.0, c, T, sigma)
    A = 4.0 * T * P * s_c ** 2 / (np.pi ** 2 * (1.0 + T)) + (2.0 * P / np.pi ** 2) * (n_d / 3.0 + n_o)
    B = (P / np.pi) * (n_d * V_d / SQRT3 + n_o * V_o)
    return A, B


def c0(P, ks, T, sigma):
    """Slope c0 of the early-time boundary c*(alpha) = c0 alpha (Sec. 4.1, App. F), at P, ks, T, sigma."""
    eps_bar = (2.0 * P / np.pi) * np.arcsin(T / (1.0 + T)) + sigma ** 2
    beta = (2.0 / (3.0 * ks)) * (eps_bar + (P - 1) / 3.0 + (2.0 / np.pi) * np.arcsin(0.25)) \
        + (1.0 - 1.0 / ks) * (eps_bar + (P - 2) / 3.0 + (4.0 / np.pi) * np.arcsin(1.0 / 3.0))
    return (1.0 + 2.0 * T / (1.0 + T) - 2.0 / (3.0 * ks)) ** 2 / (np.pi * beta)


def floor_cnn(P, c, T):
    """Small-eta noise floor of the CNN divided by eta sigma^2 (Result 4.2), at P, c, T (arrays allowed)."""
    return (P / np.sqrt(1.0 + 2.0 * T) + P * (P - 1) * c / np.sqrt((1.0 + T) ** 2 - T ** 2 * c ** 2)) / np.pi


def floor_lcn(P, T):
    """Small-eta noise floor of the LCN divided by eta sigma^2 (Result 4.2), at P, T (arrays allowed)."""
    return P / (np.pi * np.sqrt(1.0 + 2.0 * T))
