from __future__ import annotations

import math

import numpy as np


def _student_t_logpdf(x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    df = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    scale2 = np.maximum(scale2, 1e-12)
    z2 = (x - mu) ** 2 / scale2
    lgamma_vec = np.vectorize(math.lgamma)
    return (
        lgamma_vec((df + 1.0) / 2.0)
        - lgamma_vec(df / 2.0)
        - 0.5 * (np.log(df * math.pi) + np.log(scale2))
        - ((df + 1.0) / 2.0) * np.log1p(z2 / df)
    )


def bocd_change_probability(
    values: np.ndarray,
    hazard_bars: int = 40,
    max_run_length: int = 128,
) -> np.ndarray:
    """Bayesian online changepoint probabilities for a causal standardized series.

    Uses a Normal-Inverse-Gamma conjugate predictive model and a constant hazard.
    No future observation is used at any step.
    """
    if hazard_bars < 2:
        raise ValueError("hazard_bars must be >= 2")
    if max_run_length < 2:
        raise ValueError("max_run_length must be >= 2")

    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan)
    h = 1.0 / float(hazard_bars)

    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.5, 1.0
    run_p = np.array([1.0])
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    for t, value in enumerate(x):
        if not np.isfinite(value):
            continue

        log_pred_growth = _student_t_logpdf(value, mu, kappa, alpha, beta)
        log_pred_prior = float(_student_t_logpdf(
            value,
            np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])
        )[0])

        growth = run_p * (1.0 - h) * np.exp(log_pred_growth - np.max(log_pred_growth))
        cp = run_p.sum() * h * math.exp(log_pred_prior - float(np.max(log_pred_growth)))
        new_p = np.concatenate([[cp], growth])
        if len(new_p) > max_run_length + 1:
            new_p = new_p[: max_run_length + 1]
        total = new_p.sum()
        if not np.isfinite(total) or total <= 0:
            new_p = np.zeros_like(new_p)
            new_p[0] = 1.0
        else:
            new_p /= total
        out[t] = new_p[0]

        def update(m, k, a, b):
            nk = k + 1.0
            nm = (k * m + value) / nk
            na = a + 0.5
            nb = b + k * (value - m) ** 2 / (2.0 * nk)
            return nm, nk, na, nb

        pm, pk, pa, pb = update(
            np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])
        )
        gm, gk, ga, gb = update(mu, kappa, alpha, beta)
        mu = np.concatenate([pm, gm])[: len(new_p)]
        kappa = np.concatenate([pk, gk])[: len(new_p)]
        alpha = np.concatenate([pa, ga])[: len(new_p)]
        beta = np.concatenate([pb, gb])[: len(new_p)]
        run_p = new_p

    return out
