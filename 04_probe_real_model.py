"""
04 — run the S (+) A probe on a REAL pretrained transformer.

The question from the ledger:
    Does ||A|| (the skew / "arrow" half of the lag covariance) spike on
    STRUCTURED input (prose, code, math) and go flat on SCRAMBLED tokens,
    and does it PEAK in the MIDDLE layers?

Method note (this is the load-bearing methodological care):
    Raw ||A|| is NOT comparable across layers or inputs — GPT-2's activation
    scale grows several-fold with depth, so a raw ||A|| profile mostly measures
    "how big are the activations here", not "how much arrow is here". The
    scale-free quantity — the one probe.run_falsification itself endorses — is

        shuffle_ratio = ||A||(ordered)  /  ||A||(time-shuffled)

    Shuffling uses the SAME hidden states (same scale), just permuted in time,
    so the ratio isolates directed temporal structure and divides the scale out.
    That is the headline metric here. arrow_ratio = ||A||/||S|| and top_island
    are reported alongside for transparency.

Three conditions per layer, to separate the two SOURCES of an arrow
(genuine content order vs. the model's positional encoding):
    [ordered]    structured text, hidden states in their true order
                 -> arrow from BOTH content order and position
    [scrambled]  the SAME text with its INPUT TOKENS permuted before the forward
                 -> arrow from POSITION ONLY (content order destroyed)
    [floor]      time-shuffle of the ordered hidden states (= the shuffle_ratio
                 denominator) -> the no-order baseline, ratio == 1 by construction

Prediction, stated up front so it can fail:
    A) shuffle_ratio(ordered, structured)  >  shuffle_ratio(scrambled)  >~ 1.0
       i.e. real content order adds arrow on top of whatever position contributes,
       and pure token-scramble collapses most of it.
    B) the ordered structured shuffle_ratio profile PEAKS in the middle layers,
       not at the embedding layer or the final layer.

Run it where you have network + torch + transformers:
    pip install torch transformers
    python experiments/04_probe_real_model.py                 # gpt2, tau sweep {1,2,4,8,16}
    python experiments/04_probe_real_model.py gpt2-medium      # 24 layers, clearer profile
    python experiments/04_probe_real_model.py gpt2 1,2,4,8,16,32   # custom lag list

Lag sweep, because tau=1 is the lag where a locally-smooth trajectory suppresses the
arrow hardest: if a real directed flow exists (induction, positional drift) it more
plausibly lives at longer lags. The sweep is the decisive control — if NO lag lifts
the ordered arrow above the time-shuffle floor with a mid-stack peak, the premise is
dead on real models; if one does, that is the first real-model evidence for A.

Writes probe_real_<model>.csv (long form, with a tau column) and prints a per-lag
summary, per-layer detail for tau=1 and the strongest lag, and an overall verdict.

Do not hype. Do not lie. Just show.
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ember.probe import s_a_decompose


# ----------------------------------------------------------------------
# scale-free arrow metric helpers (a_norm only, no eig -> fast shuffles)
# ----------------------------------------------------------------------
def _a_norm(H, tau=1):
    H = np.asarray(H, dtype=np.float64)
    H = H - H.mean(axis=0, keepdims=True)
    T = H.shape[0]
    C = (H[:-tau].T @ H[tau:]) / (T - tau)
    A = 0.5 * (C - C.T)
    return float(np.linalg.norm(A))


def shuffle_ratio(H, tau=1, n_shuffle=16, seed=0):
    """ordered ||A|| / mean time-shuffled ||A||.  Scale-free. Floor ~= 1.0."""
    rng = np.random.default_rng(seed)
    a0 = _a_norm(H, tau)
    sh = np.array([_a_norm(H[rng.permutation(H.shape[0])], tau)
                   for _ in range(n_shuffle)])
    return a0 / (sh.mean() + 1e-12)


# ----------------------------------------------------------------------
# real-model harvesting (load ONCE, return ALL layers in one forward)
# ----------------------------------------------------------------------
class RealModel:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        self.name = model_name

    def ids(self, text, max_tokens=256):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
        return enc

    def all_layers(self, input_ids_dict):
        """returns list of (T, D) arrays, one per hidden_states entry
        (index 0 = embeddings, then one per block)."""
        with self.torch.no_grad():
            out = self.model(**input_ids_dict)
        return [hs[0].numpy() for hs in out.hidden_states]


def scramble_ids(enc, seed=0):
    """permute the input token ids in place (keeps T fixed). For GPT-2 there are
    no special tokens, so every position is content -> a clean content-order kill."""
    import torch
    rng = np.random.default_rng(seed)
    ids = enc["input_ids"].clone()
    T = ids.shape[1]
    perm = rng.permutation(T)
    ids[0] = ids[0][torch.as_tensor(perm)]
    out = {k: v.clone() for k, v in enc.items()}
    out["input_ids"] = ids
    return out


# ----------------------------------------------------------------------
# the text battery (kept long enough that T >= ~48 tokens each)
# ----------------------------------------------------------------------
PROSE = [
    ("prose",
     "The harbor town woke slowly under a thin grey light. Fishermen stacked "
     "their nets along the quay while gulls wheeled overhead, screaming for "
     "scraps. A woman in a red coat walked the length of the pier and stopped "
     "at the very end, watching the tide pull the last of the night out to sea. "
     "She had come here every morning since the accident, as if the water might "
     "one day give back what it had taken, though she knew it never would."),
    ("prose",
     "History does not repeat itself, but it often rhymes. Empires rise on the "
     "back of cheap energy and disciplined institutions, expand until their "
     "frontiers cost more to hold than they return, and then contract through a "
     "long sequence of small concessions that no single generation experiences "
     "as collapse. The people living through the decline rarely name it; they "
     "are too busy adapting to each new normal to notice the trend."),
]
CODE = [
    ("code",
     "def quicksort(xs):\n"
     "    if len(xs) <= 1:\n"
     "        return xs\n"
     "    pivot = xs[len(xs) // 2]\n"
     "    left = [x for x in xs if x < pivot]\n"
     "    mid = [x for x in xs if x == pivot]\n"
     "    right = [x for x in xs if x > pivot]\n"
     "    return quicksort(left) + mid + quicksort(right)\n"
     "\n"
     "class RingBuffer:\n"
     "    def __init__(self, n):\n"
     "        self.buf = [None] * n\n"
     "        self.i = 0\n"
     "    def push(self, x):\n"
     "        self.buf[self.i % len(self.buf)] = x\n"
     "        self.i += 1\n"),
    ("code",
     "import numpy as np\n"
     "def attention(Q, K, V):\n"
     "    scores = Q @ K.T / np.sqrt(Q.shape[-1])\n"
     "    scores = scores - scores.max(axis=-1, keepdims=True)\n"
     "    w = np.exp(scores)\n"
     "    w = w / w.sum(axis=-1, keepdims=True)\n"
     "    return w @ V\n"
     "for epoch in range(epochs):\n"
     "    loss = train_step(model, batch)\n"
     "    if epoch % 10 == 0:\n"
     "        print(epoch, loss)\n"),
]
MATH = [
    ("math",
     "Let f be continuous on the closed interval from a to b and differentiable "
     "on the open interval. Then there exists a point c strictly between a and b "
     "such that f prime of c equals f of b minus f of a divided by b minus a. "
     "Proof: define g of x equals f of x minus the secant line through the "
     "endpoints. Then g of a equals g of b, so by Rolle's theorem g prime "
     "vanishes somewhere in the interior, which gives the claim after "
     "rearranging the terms."),
    ("math",
     "The sum from n equals one to infinity of one over n squared converges to "
     "pi squared over six. Consider the Fourier series of the function x squared "
     "on the interval from minus pi to pi. Integrate term by term, evaluate the "
     "coefficients, and set x equal to pi. The cross terms cancel by symmetry "
     "and the remaining series, after dividing by four, yields the stated value "
     "of the basel sum."),
]
BATTERY = PROSE + CODE + MATH


# ----------------------------------------------------------------------
def _profile_at_tau(Hs_ord, Hs_scr, struct_ti, CATS, n_layers, tau, n_shuffle):
    """Compute, for one lag tau, the per-layer pooled-structured shuffle_ratio
    profiles (ordered + scrambled), the per-text mid-band values, and the
    ordered arrow_ratio / top-island profiles. Returns a dict."""
    sr_o = {ti: {} for ti in struct_ti}
    sr_s = {ti: {} for ti in struct_ti}
    ar = {ti: {} for ti in struct_ti}
    isl = {ti: {} for ti in struct_ti}
    for ti in struct_ti:
        for L in range(n_layers):
            sr_o[ti][L] = shuffle_ratio(Hs_ord[ti][L], tau, n_shuffle, seed=L * 100 + tau)
            sr_s[ti][L] = shuffle_ratio(Hs_scr[ti][L], tau, n_shuffle, seed=L * 100 + tau)
            d = s_a_decompose(Hs_ord[ti][L], tau)
            ar[ti][L] = d["arrow_ratio"]; isl[ti][L] = d["top_island"]

    def pool(tbl, L):
        return float(np.mean([tbl[ti][L] for ti in struct_ti]))

    prof_ord = [pool(sr_o, L) for L in range(n_layers)]
    prof_scr = [pool(sr_s, L) for L in range(n_layers)]
    prof_ar = [pool(ar, L) for L in range(n_layers)]
    prof_isl = [pool(isl, L) for L in range(n_layers)]

    mid_band = list(range(max(1, n_layers // 4), max(2, (3 * n_layers) // 4) + 1))
    struct_mid = float(np.mean([prof_ord[L] for L in mid_band]))
    scr_mid = float(np.mean([prof_scr[L] for L in mid_band]))
    lift = struct_mid / (scr_mid + 1e-12)
    peak_L = int(np.argmax(prof_ord))
    edge = max(prof_ord[0], prof_ord[-1])
    pt_ord = {ti: float(np.mean([sr_o[ti][L] for L in mid_band])) for ti in struct_ti}
    pt_scr = {ti: float(np.mean([sr_s[ti][L] for L in mid_band])) for ti in struct_ti}
    wins = sum(1 for ti in struct_ti if pt_ord[ti] > pt_scr[ti])
    n_txt = len(struct_ti)
    A = (lift >= 1.10) and (wins >= int(np.ceil(0.8 * n_txt)))
    B = (peak_L in mid_band) and (prof_ord[peak_L] >= 1.05 * edge)
    above_floor = max(prof_ord)  # does the ordered arrow clear the shuffle floor anywhere?
    return dict(tau=tau, prof_ord=prof_ord, prof_scr=prof_scr, prof_ar=prof_ar,
                prof_isl=prof_isl, sr_o=sr_o, sr_s=sr_s, ar=ar, isl=isl,
                mid_band=mid_band,
                struct_mid=struct_mid, scr_mid=scr_mid, lift=lift, peak_L=peak_L,
                edge=edge, wins=wins, n_txt=n_txt, A=A, B=B, above_floor=above_floor)


def _print_detail(R, struct_ti, CATS, n_layers):
    """Full per-layer table + ascii profile for one lag's result dict R."""
    tau = R["tau"]; prof_ord = R["prof_ord"]; prof_scr = R["prof_scr"]
    print("\n" + "-" * 70)
    print(f"PER-LAYER PROFILE @ tau={tau}  "
          f"(shuffle_ratio = ordered ||A|| / time-shuffled ||A||)")
    print(f"{'layer':>5} | {'ord(struct)':>11} {'scrambled':>10} | "
          f"{'prose':>6} {'code':>6} {'math':>6} | {'arrow_r':>7} {'island':>9}")

    def cat_mean(srtbl, cat, L):
        v = [srtbl[ti][L] for ti in struct_ti if CATS[ti] == cat]
        return float(np.mean(v)) if v else float("nan")

    for L in range(n_layers):
        p = cat_mean(R["sr_o"], "prose", L); c = cat_mean(R["sr_o"], "code", L)
        m = cat_mean(R["sr_o"], "math", L)
        print(f"{L:>5} | {prof_ord[L]:>11.3f} {prof_scr[L]:>10.3f} | "
              f"{p:>6.3f} {c:>6.3f} {m:>6.3f} | {R['prof_ar'][L]:>7.3f} {R['prof_isl'][L]:>9.2f}")
    print(f"\nordered-structured shuffle_ratio by layer @ tau={tau} (■ = bar; "
          f"dashed line = 1.0 floor):")
    lo, hi = min(prof_ord), max(prof_ord); rng_ = (hi - lo) or 1.0
    for L, v in enumerate(prof_ord):
        bar = "■" * int(1 + 40 * (v - lo) / rng_)
        flag = "  <- peak" if v == hi else ("  (>floor)" if v > 1.0 else "")
        print(f"  L{L:>2} {v:5.2f} {bar}{flag}")


def run(model_name="gpt2", lags=(1, 2, 4, 8, 16), n_shuffle=12, max_tokens=256):
    print("=" * 70)
    print(f"EXPERIMENT 04  —  S(+)A probe on a real model: {model_name}")
    print(f"                  LAG SWEEP tau in {tuple(lags)}")
    print("=" * 70)
    t0 = time.time()
    rm = RealModel(model_name)
    enc0 = rm.ids(BATTERY[0][1], max_tokens)
    n_layers = len(rm.all_layers(enc0))
    print(f"loaded {model_name}: {n_layers} hidden-state layers "
          f"(0=embeddings .. {n_layers-1}=final)\n")

    # harvest hidden states ONCE per text; reuse across all lags
    Hs_ord, Hs_scr, CATS = {}, {}, {}
    for ti, (cat, text) in enumerate(BATTERY):
        CATS[ti] = cat
        enc = rm.ids(text, max_tokens)
        Hs_ord[ti] = rm.all_layers(enc)
        Hs_scr[ti] = rm.all_layers(scramble_ids(enc, seed=ti))
        T = Hs_ord[ti][0].shape[0]
        print(f"  [{cat:5s}] text {ti}: T={T} tokens")
        if T <= max(lags) + 8:
            print(f"    (warning: T={T} is short for tau up to {max(lags)})")
    struct_ti = [ti for ti, c in CATS.items() if c in ("prose", "code", "math")]

    # run the sweep
    results = {}
    print("\n" + "-" * 70)
    print("LAG SWEEP SUMMARY  (mid-band pooled over prose/code/math)")
    print(f"{'tau':>4} | {'ord_mid':>7} {'scr_mid':>7} {'lift':>6} | "
          f"{'peakL':>5} {'peakval':>7} {'maxfloor':>8} | {'wins':>5} | {'A':>2} {'B':>2}")
    for tau in lags:
        R = _profile_at_tau(Hs_ord, Hs_scr, struct_ti, CATS, n_layers, tau, n_shuffle)
        results[tau] = R
        print(f"{tau:>4} | {R['struct_mid']:>7.3f} {R['scr_mid']:>7.3f} {R['lift']:>6.2f} | "
              f"{R['peak_L']:>5} {R['prof_ord'][R['peak_L']]:>7.3f} {R['above_floor']:>8.3f} | "
              f"{R['wins']:>2}/{R['n_txt']:<2} | "
              f"{'V' if R['A'] else 'K':>2} {'V' if R['B'] else 'K':>2}")

    # detail for tau=1 (continuity with the committed result) and the strongest lag
    best_tau = max(lags, key=lambda t: (results[t]["A"] and results[t]["B"],
                                        results[t]["lift"],
                                        results[t]["above_floor"]))
    _print_detail(results[lags[0]], struct_ti, CATS, n_layers)
    if best_tau != lags[0]:
        _print_detail(results[best_tau], struct_ti, CATS, n_layers)

    # overall verdict across the sweep
    any_transfer = [t for t in lags if results[t]["A"] and results[t]["B"]]
    # a "partial" needs a REAL mid-stack bump above the shuffle floor (margin >=1.10,
    # geometric B satisfied) — but content order not clearly beating scramble (A false).
    # the 1.10 margin keeps random ~1.02 shuffle wobble from counting as signal.
    any_above_floor = [t for t in lags
                       if results[t]["B"] and results[t]["above_floor"] >= 1.10
                       and not results[t]["A"]]
    print("\n" + "-" * 70)
    print("OVERALL VERDICT (across the lag sweep)")
    print("  NOTE: shuffle_ratio < 1 means random time-pairings carry MORE skew than the")
    print("        true order — a smooth-trajectory signature, not just absence of signal.")
    if any_transfer:
        print(f"  [V] the A-premise TRANSFERS at tau = {any_transfer}: "
              f"content-order arrow beats scramble AND peaks mid-stack.")
        print(f"      -> first real-model evidence for A. Strongest lag: tau={best_tau} "
              f"(lift {results[best_tau]['lift']:.2f}x, peak L{results[best_tau]['peak_L']}).")
    elif any_above_floor:
        print(f"  [B/PARTIAL] no lag passes BOTH tests, but at tau = {any_above_floor} the "
              f"ordered arrow clears the shuffle floor with a mid-stack peak.")
        print("      -> a real directed component exists at longer lag, but it is not clearly")
        print("         content-order-specific (scramble keeps pace). Worth a closer look.")
    else:
        print("  [K] the A-premise DOES NOT TRANSFER at any tested lag: the ordered arrow")
        print("      never clears the time-shuffle floor mid-stack, and content order never")
        print("      beats token-scramble. On a real transformer, lag-covariance skew is not")
        print("      where sequence order lives — at least not for GPT-2 on this battery.")
    print("=" * 70)

    # csv (long form, with tau)
    out = f"probe_real_{model_name.replace('/', '_')}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "layer", "cond", "text_index", "category",
                    "shuffle_ratio", "arrow_ratio", "top_island"])
        for tau in lags:
            R = results[tau]
            for ti in struct_ti:
                for L in range(n_layers):
                    w.writerow([tau, L, "ordered", ti, CATS[ti],
                                f"{R['sr_o'][ti][L]:.5f}",
                                f"{R['ar'][ti][L]:.5f}", f"{R['isl'][ti][L]:.5f}"])
                    w.writerow([tau, L, "scrambled", ti, CATS[ti],
                                f"{R['sr_s'][ti][L]:.5f}", "", ""])
    print(f"\nwrote {out}   (elapsed {time.time()-t0:.1f}s)")
    print("send me the LAG SWEEP SUMMARY table + the OVERALL VERDICT and I'll fold it in.")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    # optional: pass lags as a comma list, e.g.  python 04_...py gpt2 1,2,4,8,16,32
    lags = (tuple(int(x) for x in sys.argv[2].split(","))
            if len(sys.argv) > 2 else (1, 2, 4, 8, 16))
    run(model, lags=lags)