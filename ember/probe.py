"""
The S (+) A probe.

Parameter-free. Give it the hidden states H (T, D) a model produced for one
sequence and it splits the time-lagged covariance into its symmetric and skew
halves:

    C_tau = (1/(T-tau)) * sum_t H[t]^T H[t+tau]
    S = (C_tau + C_tau^T)/2    the time-blind power half  ("the dictionary")
    A = (C_tau - C_tau^T)/2    the directed/rotation half ("the arrow")

It returns:
    arrow_ratio = ||A|| / ||S||          how much of the lagged structure is directed
    islands     = sorted +imag eigvals of A   the rotation frequencies ("spectral islands")

A real skew-symmetric matrix has purely imaginary eigenvalues in +/- conjugate
pairs; each pair is one 2D invariant plane. That is the same object that, in the
GeometricNeuron line, the Chiral Eye reads and V9 recovers as emergent islands.

The headline falsification test (run_falsification): if A genuinely carries
sequence order, arrow_ratio must be HIGHER on an ordered sequence than on the same
sequence with its time steps shuffled. If shuffling does not collapse the arrow,
the premise is weaker than claimed.
"""
import numpy as np


def s_a_decompose(H, tau=1):
    H = np.asarray(H, dtype=np.float64)
    H = H - H.mean(axis=0, keepdims=True)
    T = H.shape[0]
    C = (H[:-tau].T @ H[tau:]) / (T - tau)
    S = 0.5 * (C + C.T)
    A = 0.5 * (C - C.T)
    a_norm = float(np.linalg.norm(A))
    s_norm = float(np.linalg.norm(S))
    eig = np.linalg.eigvals(A)
    islands = np.sort(eig.imag[eig.imag > 1e-9])[::-1]
    top_island = float(islands[0]) if islands.size else 0.0
    return {"a_norm": a_norm, "s_norm": s_norm,
            "arrow_ratio": a_norm / (s_norm + 1e-12),
            "top_island": top_island, "islands": islands, "S": S, "A": A}


def run_falsification(H, tau=1, n_shuffle=20, seed=0):
    """Compare ABSOLUTE A-structure on ordered H vs time-shuffled H.

    arrow_ratio = ||A||/||S|| is NOT the right metric: shuffling collapses the
    whole lag covariance to noise, where ||A|| ~ ||S|| (ratio -> ~1), while an
    ordered signal has a large symmetric autocorrelation that DEPRESSES the ratio.
    The order lives in the ABSOLUTE size of A (and its top island), which must
    fall when the time axis is shuffled.
    """
    rng = np.random.default_rng(seed)
    real = s_a_decompose(H, tau)
    sa, si = [], []
    for _ in range(n_shuffle):
        idx = rng.permutation(H.shape[0])
        d = s_a_decompose(H[idx], tau)
        sa.append(d["a_norm"]); si.append(d["top_island"])
    sa, si = np.array(sa), np.array(si)
    return {"ordered_anorm": real["a_norm"], "shuffled_anorm": float(sa.mean()),
            "anorm_ratio": float(real["a_norm"] / (sa.mean() + 1e-12)),
            "ordered_island": real["top_island"], "shuffled_island": float(si.mean()),
            "island_ratio": float(real["top_island"] / (si.mean() + 1e-12))}


def harvest_hidden_states(model_name, text, layer=6):
    """
    Convenience hook for running the probe on a REAL pretrained transformer.
    Not exercised in the offline test suite (needs a model download). Run it
    yourself where you have network + transformers installed:

        from ember.probe import harvest_hidden_states, s_a_decompose
        H = harvest_hidden_states("gpt2", "the cat sat on the mat ...", layer=6)
        print(s_a_decompose(H)["arrow_ratio"])
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    with torch.no_grad():
        ids = tok(text, return_tensors="pt")
        out = model(**ids)
    return out.hidden_states[layer][0].numpy()  # (T, D)
