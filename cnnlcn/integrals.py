"""Gaussian integrals for g(z) = erf(z / sqrt 2) (Appendix A), and the ReLU kernel.

Each function takes entries c_ij of the covariance matrix of zero-mean Gaussian
variables u_1, u_2, ... as arrays of any common broadcast shape and returns an array
of that shape.
"""
import numpy as np

TWO_OVER_PI = 2.0 / np.pi


def _asin(z):
    """arcsin of z (array), with z clipped to [-1, 1] against rounding."""
    return np.arcsin(np.clip(z, -1.0, 1.0))


def I2(c11, c12, c22):
    """E[g(u1) g(u2)] from the covariance entries c11, c12, c22 (arrays). Returns an array."""
    return TWO_OVER_PI * _asin(c12 / np.sqrt((1.0 + c11) * (1.0 + c22)))


def J2(c11, c12, c22):
    """E[g'(u1) g'(u2)] from the covariance entries c11, c12, c22 (arrays). Returns an array."""
    return TWO_OVER_PI / np.sqrt((1.0 + c11) * (1.0 + c22) - c12 ** 2)


def I3(c11, c12, c13, c23, c33):
    """E[g'(u1) u2 g(u3)] from the covariance entries (arrays). Returns an array.

    The variance of u2 does not enter.
    """
    return TWO_OVER_PI * (c23 * (1.0 + c11) - c12 * c13) / (
        (1.0 + c11) * np.sqrt((1.0 + c11) * (1.0 + c33) - c13 ** 2))


def I4(c11, c12, c13, c14, c22, c23, c24, c33, c34, c44):
    """E[g'(u1) g'(u2) g(u3) g(u4)] from the ten covariance entries (arrays). Returns an array."""
    L4 = (1.0 + c11) * (1.0 + c22) - c12 ** 2
    L1 = L4 * (1.0 + c33) - c23 ** 2 * (1.0 + c11) - c13 ** 2 * (1.0 + c22) + 2.0 * c12 * c13 * c23
    L2 = L4 * (1.0 + c44) - c24 ** 2 * (1.0 + c11) - c14 ** 2 * (1.0 + c22) + 2.0 * c12 * c14 * c24
    L0 = (L4 * c34 - c23 * c24 * (1.0 + c11) - c13 * c14 * (1.0 + c22)
          + c12 * c13 * c24 + c12 * c14 * c23)
    return (4.0 / np.pi ** 2) / np.sqrt(L4) * _asin(L0 / np.sqrt(L1 * L2))


def I2_relu(c11, c12, c22):
    """E[relu(u1) relu(u2)] from c11, c12, c22 (arrays), the arc-cosine kernel. Returns an array.

    Used for the ReLU figures of Appendix J.
    """
    ab = np.maximum(c11 * c22, 1e-300)
    return (np.sqrt(np.maximum(ab - c12 ** 2, 0.0))
            + c12 * (np.pi - np.arccos(np.clip(c12 / np.sqrt(ab), -1.0, 1.0)))) / (2.0 * np.pi)
