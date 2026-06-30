"""
GRU baseline. A standard stateful recurrent cell — the external reference for
"a competent sequence model that pays uniform compute at every step." Sized to
match Ember's parameter count.
"""
import autograd.numpy as anp
import numpy as np
from .optim import sigmoid


def init_params(obs_dim=4, H=38, seed=0):
    rng = np.random.default_rng(seed)

    def W(a, b):
        return rng.normal(0, 1.0 / np.sqrt(a), (b, a))
    return {
        "Wz": W(obs_dim, H), "Uz": W(H, H), "bz": np.zeros(H),
        "Wr": W(obs_dim, H), "Ur": W(H, H), "br": np.zeros(H),
        "Wh": W(obs_dim, H), "Uh": W(H, H), "bh": np.zeros(H),
        "W_dec": W(H, obs_dim), "b_dec": np.zeros(obs_dim),
        "W_task": W(H, 1), "b_task": np.zeros(1),
    }


def forward(p, X):
    B, T, _ = X.shape
    H = p["bz"].shape[0]
    h = anp.zeros((B, H))
    preds, tasks = [], []
    xhat = anp.dot(h, p["W_dec"].T) + p["b_dec"]
    for t in range(T):
        xt = X[:, t, :]
        z = sigmoid(anp.dot(xt, p["Wz"].T) + anp.dot(h, p["Uz"].T) + p["bz"])
        r = sigmoid(anp.dot(xt, p["Wr"].T) + anp.dot(h, p["Ur"].T) + p["br"])
        hh = anp.tanh(anp.dot(xt, p["Wh"].T) + anp.dot(r * h, p["Uh"].T) + p["bh"])
        h = (1 - z) * h + z * hh
        xhat = anp.dot(h, p["W_dec"].T) + p["b_dec"]
        preds.append(xhat)
        tasks.append(anp.dot(h, p["W_task"].T) + p["b_task"])
    return {"preds": anp.stack(preds, axis=1), "task": anp.stack(tasks, axis=1)}


def next_step_loss(p, X):
    preds = forward(p, X)["preds"]
    return anp.mean((preds[:, :-1, :] - X[:, 1:, :]) ** 2)


def parity_loss(p, X, Y, aux=0.3):
    out = forward(p, X)
    pp = sigmoid(out["task"])
    bce = -anp.mean(Y * anp.log(pp + 1e-7) + (1 - Y) * anp.log(1 - pp + 1e-7))
    carrier = anp.mean((out["preds"][:, :-1] - X[:, 1:]) ** 2)
    return bce + aux * carrier
