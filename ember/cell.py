"""
The Ember cell.

One recurrent unit that factors its interaction into two halves and pays for
them differently:

    S = (Msym + Msymᵀ)/2   the CHEAP held half — a symmetric associative settle.
                           Always runs. The "transformer-like" content match.
    A = (Mskew - Mskewᵀ)/2 the DEAR moving half — a skew (rotation) flow.
                           Runs ONLY through an excitable gate driven by surprise.

The gate g_t in [0,1] is the FitzHugh-Nagumo-style symmetry breaker (V13->V14 in
the GeometricNeuron line: you cannot DRAW an arrow into passive geometry, you must
break a symmetry to MAKE one). It opens on prediction error and is suppressed by a
refractory trace r_t, so firing is history-dependent — which is what makes the
A-flow non-reciprocal rather than a reversible linear rotation.

theta/rotor = the PROTECTED LATCH. A 2D rotor per register, init (1,0). A detected
surprise commits a fixed ~pi flip; quiet steps hold. Hold a rotor and it does not
drift; the arrow stays dark and the bit is kept for free. (Status: the detector
half works well; the latch is brittle on long horizons — see the ledger.)

Two readout heads:
    carrier head  predicts the next observation (used to compute surprise)
    task head     reads the protected memory (h + rotor) for the actual task

Everything is batched: state h is (B, D), input x is (B, obs_dim).
"""
import autograd.numpy as anp
import numpy as np
from .optim import sigmoid


def init_params(obs_dim=4, D=48, K=4, seed=0, skew_scale=0.4):
    rng = np.random.default_rng(seed)
    s = 1.0 / np.sqrt(D)
    return {
        "W_in":  rng.normal(0, s, (D, obs_dim)),
        "b":     np.zeros(D),
        "Msym":  rng.normal(0, s, (D, D)),
        "Mskew": rng.normal(0, skew_scale * s, (D, D)),
        # carrier prediction head (for surprise): h -> next obs
        "W_dec": rng.normal(0, s, (obs_dim, D)),
        "b_dec": np.zeros(obs_dim),
        # task head: [h, sin theta, cos theta] -> 1 logit
        "W_task": rng.normal(0, 1.0 / np.sqrt(D + 2 * K), (1, D + 2 * K)),
        "b_task": np.zeros(1),
        "omega_w": rng.uniform(2.7, 3.3, K),   # sets register count K; rotor flip is fixed pi
        "g_logslope": np.array(2.0),
        "g_thresh":   np.array(-1.5),
        "g_refleak":  np.array(1.0),
    }


def forward(p, X, gate_on=True, dt=0.5):
    """
    X: (B, T, obs_dim)
    gate_on=False forces g_t = 1 everywhere (the ablation that pays uniformly and
    advances the winding every step — which should break protected memory).
    Returns preds (carrier), task (parity logits), gate, aflow.
    """
    B, T, _ = X.shape
    D = p["b"].shape[0]
    K = p["omega_w"].shape[0]
    S = 0.5 * (p["Msym"] + p["Msym"].T)
    A = 0.5 * (p["Mskew"] - p["Mskew"].T)
    slope = anp.exp(p["g_logslope"])
    rho = sigmoid(p["g_refleak"])

    h = anp.zeros((B, D))
    zx = anp.ones((B, K))      # protected rotor latch, init (1,0) per register
    zy = anp.zeros((B, K))
    r = anp.zeros(B)
    xhat = anp.dot(h, p["W_dec"].T) + p["b_dec"]   # predicts X[:,0]

    preds, tasks, gates, aflows = [], [], [], []
    for t in range(T):
        xt = X[:, t, :]
        err = anp.mean((xt - xhat) ** 2, axis=1)
        if gate_on:
            g = sigmoid(slope * (err - p["g_thresh"])) * (1.0 - r)
            r = rho * r + (1.0 - rho) * g
        else:
            g = anp.ones(B)
        gates.append(g)

        u = anp.dot(xt, p["W_in"].T)
        h_sym = h + dt * (-h + anp.dot(anp.tanh(h), S.T) + u + p["b"])   # cheap settle
        flow = anp.dot(h_sym, A.T)                                       # the arrow
        h = h_sym + dt * g[:, None] * flow                              # gated
        # protected rotor latch (flip-flop): a DETECTED surprise commits a fixed
        # ~pi flip; quiet steps hold. Commitment is decoupled from the exact gate
        # magnitude so partial gate values still produce clean flips.
        commit = sigmoid(8.0 * (g - 0.5))
        ang = (np.pi * commit)[:, None] * anp.ones((1, K))
        c, sn = anp.cos(ang), anp.sin(ang)
        zx, zy = zx * c - zy * sn, zx * sn + zy * c
        aflows.append(anp.sqrt(anp.mean(flow ** 2, axis=1) + 1e-12) * g)

        feat = anp.concatenate([h, zx, zy], axis=1)
        tasks.append(anp.dot(feat, p["W_task"].T) + p["b_task"])       # parity logit
        xhat = anp.dot(h, p["W_dec"].T) + p["b_dec"]                    # predicts X[:,t+1]
        preds.append(xhat)

    return {
        "preds": anp.stack(preds, axis=1),
        "task":  anp.stack(tasks, axis=1),     # (B,T,1) logits
        "gate":  anp.stack(gates, axis=1),
        "aflow": anp.stack(aflows, axis=1),
    }


def parity_loss(p, X, Y, gate_on=True, aux=0.3, lam=0.02):
    """
    Total loss = parity BCE  +  aux * carrier next-step MSE  +  lam * mean(gate).

    The carrier term keeps the surprise signal meaningful (predict the smooth part
    well so resets stand out). The lam term is the PRICE OF THE ARROW — the Still/
    Crooks economy made explicit: spending the gate costs, so the cell is pushed to
    fire only where it pays off. lam=0 removes the price (ablation).
    """
    out = forward(p, X, gate_on=gate_on)
    logit = out["task"]                          # (B,T,1)
    pp = sigmoid(logit)
    bce = -anp.mean(Y * anp.log(pp + 1e-7) + (1 - Y) * anp.log(1 - pp + 1e-7))
    carrier = anp.mean((out["preds"][:, :-1] - X[:, 1:]) ** 2)
    gatecost = anp.mean(out["gate"])
    return bce + aux * carrier + lam * gatecost
