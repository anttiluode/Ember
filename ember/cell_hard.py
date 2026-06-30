"""
Ember cell — HARD-LATCH variant (the move flagged in README #3 / PAPER §4).

Identical to ember/cell.py in every way EXCEPT the rotor commitment:

    soft cell:   commit = sigmoid(8*(g-0.5))           # a value in [0,1]
                 -> a partial gate gives a PARTIAL rotation; sub-pi drifts
                    accumulate over the horizon and corrupt the parity bit.

    hard cell:   commit = 1{g > 0.5}  via straight-through estimator
                 -> the rotor flips a FULL pi or does not move at all. No drift.
                    Gradient still flows (soft sigmoid) so the detector trains.

This is the one-line hypothesis from the ledger: the detector is fine, the
*latch built on it* is brittle because it is soft. Make the flip discrete and
see whether parity closes the 0.63 -> 0.91 gap to the GRU.

Everything else (S settle, A flow, excitable refractory gate, heads, losses) is
copied verbatim so the comparison isolates exactly one change.
"""
import autograd.numpy as anp
import numpy as np
from autograd.core import getval as _detach
from .optim import sigmoid

# reuse init + losses' structure from the soft cell so params are identical
from .cell import init_params  # noqa: F401  (same parameter set, same seeds)


def _straight_through_commit(g):
    """forward = 1{g>0.5}; backward = d/dg sigmoid(8*(g-0.5)). Bengio STE."""
    soft = sigmoid(8.0 * (g - 0.5))
    hard = (_detach(g) > 0.5).astype(float)
    return hard + (soft - _detach(soft))   # value=hard, grad=soft


def forward(p, X, gate_on=True, force=None, dt=0.5):
    B, T, _ = X.shape
    D = p["b"].shape[0]
    K = p["omega_w"].shape[0]
    S = 0.5 * (p["Msym"] + p["Msym"].T)
    A = 0.5 * (p["Mskew"] - p["Mskew"].T)
    slope = anp.exp(p["g_logslope"])
    rho = sigmoid(p["g_refleak"])

    h = anp.zeros((B, D))
    zx = anp.ones((B, K))
    zy = anp.zeros((B, K))
    r = anp.zeros(B)
    xhat = anp.dot(h, p["W_dec"].T) + p["b_dec"]

    preds, tasks, gates, aflows = [], [], [], []
    for t in range(T):
        xt = X[:, t, :]
        err = anp.mean((xt - xhat) ** 2, axis=1)
        if force is not None:
            g = force * anp.ones(B)
        elif gate_on:
            g = sigmoid(slope * (err - p["g_thresh"])) * (1.0 - r)
            r = rho * r + (1.0 - rho) * g
        else:
            g = anp.ones(B)
        gates.append(g)

        u = anp.dot(xt, p["W_in"].T)
        h_sym = h + dt * (-h + anp.dot(anp.tanh(h), S.T) + u + p["b"])
        flow = anp.dot(h_sym, A.T)
        h = h_sym + dt * g[:, None] * flow

        # --- the ONLY change: hard straight-through flip ---
        commit = _straight_through_commit(g)              # in {0,1} on the forward pass
        ang = (np.pi * commit)[:, None] * anp.ones((1, K))
        c, sn = anp.cos(ang), anp.sin(ang)
        zx, zy = zx * c - zy * sn, zx * sn + zy * c
        # ---------------------------------------------------
        aflows.append(anp.sqrt(anp.mean(flow ** 2, axis=1) + 1e-12) * g)

        feat = anp.concatenate([h, zx, zy], axis=1)
        tasks.append(anp.dot(feat, p["W_task"].T) + p["b_task"])
        xhat = anp.dot(h, p["W_dec"].T) + p["b_dec"]
        preds.append(xhat)

    return {
        "preds": anp.stack(preds, axis=1),
        "task":  anp.stack(tasks, axis=1),
        "gate":  anp.stack(gates, axis=1),
        "aflow": anp.stack(aflows, axis=1),
    }


def parity_loss(p, X, Y, gate_on=True, aux=0.3, lam=0.02):
    out = forward(p, X, gate_on=gate_on)
    logit = out["task"]
    pp = sigmoid(logit)
    bce = -anp.mean(Y * anp.log(pp + 1e-7) + (1 - Y) * anp.log(1 - pp + 1e-7))
    carrier = anp.mean((out["preds"][:, :-1] - X[:, 1:]) ** 2)
    gatecost = anp.mean(out["gate"])
    return bce + aux * carrier + lam * gatecost
