"""Minimal Adam over a flat dict of parameters, plus param counting."""
import autograd.numpy as anp
import numpy as np


def count_params(params):
    return int(sum(np.asarray(v).size for v in params.values()))


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(np.asarray(v)) for k, v in params.items()}
        self.v = {k: np.zeros_like(np.asarray(v)) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        out = {}
        for k in params:
            g = np.asarray(grads[k])
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            out[k] = np.asarray(params[k]) - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return out


def sigmoid(x):
    return 0.5 * (anp.tanh(0.5 * x) + 1.0)
