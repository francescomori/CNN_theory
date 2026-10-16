"""Integration of the ODEs, specialization time and noise floor."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from . import model


def integrate(arch, student, y0, eta, sigma, alpha_max, n_out=300, log_grid=True, stop_eps=None,
              rtol=1e-8, atol=1e-11):
    """Integrate the ODEs of a student from y0 (LSODA).

    Inputs : arch, student ('cnn' or 'lcn'), y0 (flat state), eta, sigma, alpha_max,
             n_out (number of output times), log_grid (output times log-spaced from 1e-2,
             else linear), stop_eps (stop once eps_g falls below it, optional), rtol, atol.
    Outputs: dict with alpha (n_out,), Y (state size, n_out) and eps (n_out,).
    """
    f = lambda t, y: model.rhs(arch, student, y, eta, sigma)
    event = None
    if stop_eps is not None:
        event = lambda t, y: model.gen_error(arch, student, y) - stop_eps
        event.terminal, event.direction = True, -1
    sol = solve_ivp(f, (0.0, alpha_max), y0, method='LSODA', rtol=rtol, atol=atol,
                    dense_output=True, events=event)
    t_end = sol.t[-1]
    if log_grid:
        t = np.r_[0.0, np.logspace(-2, np.log10(max(t_end, 0.02)), n_out - 1)]
    else:
        t = np.linspace(0.0, t_end, n_out)
    Y = sol.sol(np.minimum(t, t_end))
    eps = np.array([model.gen_error(arch, student, Y[:, i]) for i in range(t.size)])
    return dict(alpha=np.minimum(t, t_end), Y=Y, eps=eps)


def spec_time(arch, y0, eta, sigma=0.0, thr=0.99, alpha_max=1e5):
    """Specialization time of the CNN: first alpha at which max_s gamma_s reaches thr (Result 3.3).

    The integration stops at the crossing (event detection).
    Inputs : arch, y0 (flat CNN state), eta, sigma, thr (threshold gamma_th), alpha_max.
    Outputs: alpha_spec (float, nan if not reached by alpha_max).
    """
    f = lambda t, y: model.rhs(arch, 'cnn', y, eta, sigma)
    event = lambda t, y: model.alignment(arch, *model.unpack(arch, 'cnn', y)).max() - thr
    event.terminal, event.direction = True, 1
    sol = solve_ivp(f, (0.0, alpha_max), y0, method='LSODA', rtol=1e-8, atol=1e-11, events=event)
    return float(sol.t_events[0][0]) if len(sol.t_events[0]) else np.nan


def noise_floor(arch, student, eta, sigma):
    """Asymptotic error at finite eta: the fixed point of the ODEs near perfect recovery.

    Found by least squares from perfect recovery. Q is stored as a full matrix, so its
    symmetry is imposed as extra equations; otherwise its antisymmetric part would be
    an undetermined direction of the fixed-point equations.
    Inputs : arch, student ('cnn' or 'lcn'), eta, sigma.
    Outputs: eps_g at the fixed point (float).
    """
    def residual(y):
        """Drift (rescaled by eta) and antisymmetric part of Q, stacked."""
        Q = model.unpack(arch, student, y)[1]
        QT = Q.T if student == 'cnn' else Q.transpose(1, 0, 3, 2)
        return np.r_[model.rhs(arch, student, y, eta, sigma) / eta, (Q - QT).ravel()]
    sol = least_squares(residual, model.perfect_recovery(arch, student), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return model.gen_error(arch, student, sol.x)
