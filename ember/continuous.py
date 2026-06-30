"""
Continuous-signal regime for testing the Ember economy.

A noisy oscillator whose hidden angular velocity omega is piecewise-constant and
switches RARELY, then persists. The model sees a noisy observation and must predict
the CLEAN next observation. That needs two things at once:

  * denoising / integration  — average out the per-step noise (a job for carried state),
  * regime tracking          — when omega switches, the carried estimate is wrong and
                               must be dumped and rebuilt (a job for a surprise event).

Between switches the stream is predictable, so a surprise-gated model can coast with
the arrow dark; at a switch it must spend. This is the regime the parity toy could
not reach and the one the efficiency claim lives or dies on.

generate_switching() returns:
    Xnoisy (n, T, obs_dim)   the input (clean + Gaussian noise)
    Xclean (n, T, obs_dim)   the target to predict
    switches (n, T)  bool, ground-truth regime-change steps

For REAL audio/video, frame your own time series with from_timeseries() and feed the
same harness — see the docstring there.
"""
import numpy as np


def generate_switching(n_seqs=512, T=96, obs_dim=4, switch_prob=0.03,
                       omega_lo=0.15, omega_hi=0.55, noise=0.30, seed=0):
    rng = np.random.default_rng(seed)
    n_harm = max(1, obs_dim // 2)
    Xc = np.zeros((n_seqs, T, obs_dim))
    sw = np.zeros((n_seqs, T), dtype=bool)
    for i in range(n_seqs):
        phi = rng.uniform(0, 2 * np.pi)
        omega = rng.uniform(omega_lo, omega_hi)
        for t in range(T):
            if t > 0 and rng.random() < switch_prob:
                omega = rng.uniform(omega_lo, omega_hi)
                sw[i, t] = True
            for h in range(n_harm):
                Xc[i, t, 2 * h] = np.cos((h + 1) * phi)
                if 2 * h + 1 < obs_dim:
                    Xc[i, t, 2 * h + 1] = np.sin((h + 1) * phi)
            phi = phi + omega
    Xn = Xc + noise * rng.normal(size=Xc.shape)
    return Xn, Xc, sw


def batches(Xn, Xc, sw, batch_size, rng):
    n = Xn.shape[0]
    while True:
        idx = rng.integers(0, n, size=batch_size)
        yield Xn[idx], Xc[idx], sw[idx]


def from_timeseries(signal, obs_dim=4, window=96, hop=48, normalize=True):
    """
    Frame a REAL 1-D (or multi-channel) signal into (n, window, obs_dim) windows for
    the same harness. Use this to point Ember at real continuous data on your machine:

        # audio envelope (e.g. librosa RMS frames, shape (T,))
        Xn = from_timeseries(rms, obs_dim=1, window=128)

        # webcam loop: per-frame features, shape (T, C) — e.g. mean optical-flow
        # vector, or a few PCA components of the frame, or your Gabor-packet codes
        Xn = from_timeseries(flow_feats, obs_dim=flow_feats.shape[1], window=96)

    There is no separate clean target for real data, so train the model to predict its
    OWN next input (self-supervised next-step); the gate will still light up on the
    unpredictable moments (scene cuts, transients, onsets). signal: (T,) or (T, C).
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if normalize:
        x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    if x.shape[1] < obs_dim:           # tile channels up to obs_dim
        x = np.tile(x, (1, int(np.ceil(obs_dim / x.shape[1]))))[:, :obs_dim]
    elif x.shape[1] > obs_dim:
        x = x[:, :obs_dim]
    wins = [x[s:s + window] for s in range(0, len(x) - window + 1, hop)]
    return np.stack(wins, axis=0) if wins else np.empty((0, window, obs_dim))
