# Ember

**A sequence cell that holds for free and pays only for surprise.**

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.

> Do not hype. Do not lie. Just show.

---

## What this is

A small, fully-runnable testbed for one idea taken out of the GeometricNeuron line
and turned into an actual trainable architecture: a recurrent cell that factors its
interaction into two halves and **pays for them differently**.

```
C_τ = S ⊕ A
S = (M+Mᵀ)/2   the CHEAP held half — a symmetric associative settle. Always runs.
A = (M−Mᵀ)/2   the DEAR moving half — a skew rotation. Runs ONLY through an
               excitable gate driven by prediction error (surprise).
```

A transformer lives almost entirely in `S` (attention is symmetric associative
retrieval) and staples its arrow on from outside as a positional coordinate. Ember
keeps `A`, **generates** it with an excitable element (you cannot draw an arrow into
passive geometry — V13→V14 in the parent line — you have to break a symmetry to make
one), and only spends compute on it when the input violates what the held state
predicted. Long-horizon memory is parked in a **protected rotor latch** (the whorl
idea): a 2D rotor that flips ~π on a detected surprise and holds for free otherwise.

This repo is not a claim that Ember beats anything. It is an honest **kill/keep**:
build the mechanism, point it at tasks designed to test it, and report what held and
what broke.

---

## The result, in one glance

Run it yourself (numbers below are from the committed code, CPU, seeds fixed):

| claim | status | number |
|---|---|---|
| `A` (skew lag-covariance) carries sequence order **on synthetic ground truth** | **[V]** | shuffling the time axis collapses ‖A‖ **5.05×**, top island **4.85×** |
| the surprise-gate makes arrow-compute track surprise | **[V]** | gate fires **3.4× more** on surprises than quiet; arrow **dark ~75–90%** of steps |
| the gate is load-bearing (not decoration) | **[V]** | removing it (gate≡1) collapses parity to chance |
| the cell solves hard long-horizon discrete memory | **[K]** | parameter-matched **GRU 0.91** vs **Ember 0.53–0.71** on 64-step parity (seed-dependent) |
| making the rotor flip **discrete** fixes the latch | **[K]** | hard straight-through flip helps **+0.07** but **kills the economy** (concentration 3.4×→**1.0×**) and still loses to GRU — *see exp 03* |
| the `A`-premise transfers to a **real trained transformer** | **[K]** | on GPT-2, swept across **τ = 1–16**, the ordered mid-stack arrow stays **below** the time-shuffle floor at every lag; content-order lift over scramble **1.01–1.06×**; the only arrow is **edge-localized and positional** — *see exp 04* |
| the protected latch reliably stores a bit | **[B/K]** | works as a detector; brittle to residual false-flips over long horizons |
| the arrow helps on a continuous tracking stream | **[V]** | arrow-on (uniform) beats arrow-off (S-only) by **~25–30%** clean-MSE |
| the gate concentrates compute on a continuous *prediction* task | **[K]** | concentration **~1.0×** — plain prediction gives no reason to fire on events |

**The honest headline:** the *economy* is real **but conditional**, and load-bearing. The
*latch* is built, partly works, and is beaten by a plain recurrent state — and exp 03 showed
that hardening the flip buys a little accuracy only by abandoning the economy, so discreteness
is not the fix; the gate is. And the *synthetic premise underneath all of it does not transfer
to a real transformer* — exp 04 swept the lag and the content-order arrow is absent in the
content-processing middle at **every** lag; the only directed structure GPT-2 carries in its
residual stream is **positional** (an embedding-layer arrow that grows with lag and is
content-independent) and a **readout** signal at the output layer. Read carefully, that *kills*
the prediction the probe was built to test while being *consistent with* Ember's motivating
contrast — that a transformer's time is a stapled-on positional label, not a generated flow.

---

## Run the experiments

```bash
pip install -r requirements.txt
python experiments/01_probe_sanity.py     # does A carry order on synthetic ground truth?
python experiments/02_kill_keep.py        # train Ember vs GRU; does the gate earn its keep?
python experiments/03_fix_the_latch.py    # soft vs HARD straight-through latch vs GRU
python experiments/04_probe_real_model.py # does the A-premise transfer to a real transformer? (lag sweep)
python continuous/efficiency.py           # does the economy pay on a continuous stream?
python continuous/economy_regime.py       # noise sweep: where (if anywhere) does it pay?
```

`04` needs `torch` + `transformers` and a one-time model download; it sweeps τ ∈ {1,2,4,8,16}
(`python experiments/04_probe_real_model.py gpt2-medium` for a deeper profile).
See [`continuous/README.md`](continuous/README.md) for the efficiency story in full.

Each prints its own numbers and its own verdict. Nothing is hidden in a figure the
print-out doesn't also state.

---

## Experiment 03 — fix the latch, in full

`cell_hard.py` is exactly `cell.py` with one change — the rotor commitment is a Bengio
straight-through estimator (forward: a full π flip or nothing; backward: the soft sigmoid
gradient, so the detector still trains). `03_fix_the_latch.py` runs it against the soft
cell and a parameter-matched GRU on parity-of-surprises.

| variant (900 steps) | parity acc | gate concentration | quiet-step rotor drift |
|---|---|---|---|
| Ember-soft | 0.714 | **3.41×** | 0.337 rad/step |
| Ember-hard | **0.783** | **1.00×** | 0.638 rad/step |
| GRU | 0.875 | — | — |

3-seed means (450 steps): soft **0.532 ± 0.013**, hard **0.587 ± 0.040**, GRU **0.912 ± 0.033**.

The hard flip improves parity over the soft latch (small but consistent across seeds) **but
does not reach GRU level, raises quiet-step drift, and collapses gate concentration
3.41×→1.00×.** It buys accuracy by letting the task head route around a no-longer-selective
latch, not by the latch holding. **Verdict: PARTIAL — discreteness is not the fix; the gate
is the seam.**

---

## Experiment 04 — run the probe on a real transformer (lag sweep)

`probe.py` decomposes any model's hidden states into `S ⊕ A` and reports the **shuffle ratio**
= ordered ‖A‖ / time-shuffled ‖A‖ (scale-free; raw ‖A‖ is confounded by GPT-2's depth-growing
activations). `04` runs it across all layers on six structured passages (prose / code / math),
against the same text with its **input tokens scrambled** (isolating positional arrow from
content-order arrow), and **sweeps the lag** τ ∈ {1,2,4,8,16} — because τ=1 is the lag where a
locally-smooth trajectory suppresses the arrow hardest, so any real directed flow should show
at longer lags if it shows anywhere.

**Lag-sweep summary (GPT-2, mid-band pooled over prose/code/math):**

| τ | ord_mid | scr_mid | lift | peak layer | peak val | wins | A | B |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.814 | 0.765 | 1.06× | L12 | 1.094 | 5/6 | K | K |
| 2 | 0.850 | 0.843 | 1.01× | L12 | 1.271 | 3/6 | K | K |
| 4 | 0.822 | 0.782 | 1.05× | L12 | 1.460 | 5/6 | K | K |
| 8 | 0.779 | 0.817 | 0.95× | L0 | 1.262 | 3/6 | K | K |
| 16 | 0.893 | 0.867 | 1.03× | L0 | 1.848 | 4/6 | K | K |

**No lag transfers.** Both tests fail at every lag: content order never beats token-scramble
(lift 1.01–1.06×, wins a coin-flip), and the peak is never mid-stack — it is always at an **edge**.

**Where the arrow actually is (ordered shuffle-ratio, edges vs middle):**

| τ | L0 (emb) | L6 (mid) | L12 (out) |
|---|---|---|---|
| 1 | 0.881 | 0.805 | 1.094 |
| 4 | 1.001 | 0.795 | 1.460 |
| 8 | 1.262 | **0.748** | 0.971 |
| 16 | **1.848** | 0.850 | 0.975 |

The content-processing **middle sits below the shuffle floor at every lag** (L6 ≈ 0.75–0.86) —
random time-pairings carry *more* skew than the true order, the smooth-trajectory signature. The
only structure above the floor is at the two ends, and it splits into two **non-Ember** sources:

- **a positional arrow** at the embedding layer that grows with lag (L0: 1.00 → 1.26 → **1.85**
  as τ goes 4 → 8 → 16) and is **provably content-independent**: at τ=8 and τ=16 the ordered and
  scrambled values are identical (lift **1.00×**, **1.01×**), because scrambling permutes tokens
  but not positions. That is the position code, nothing else.
- **a readout arrow** at the output layer (L12), content-*modulated* — at τ=1 it is code 1.31 >
  prose 1.12 > math 0.86, strongest where next-token prediction is most structured — but it is
  the prediction head at the edge, not a representational order-flow.

**Verdict: DOES NOT TRANSFER, at any lag.** The specific, testable prediction — *‖A‖ spikes on
structure and peaks mid-stack* — is dead on GPT-2 across τ = 1–16, and the mechanism is clear:
the transformer's residual-stream arrow is positional and edge-localized, not a content-order
flow. Two honest codas:

1. **This is consistent with Ember's framing, not a refutation of it.** Ember is built on the
   claim that a transformer "staples its arrow on from outside as a positional coordinate — time
   is a label, not a flow." The probe found exactly that: the only measurable arrow *is* the
   positional code. The prediction that failed was the optimistic one (a rich emergent
   content-order arrow); the contrast that motivated Ember stands.
2. **Scope.** This is a *linear, second-order* probe of the residual stream. Sequence order in a
   transformer is carried by **attention routing** — content-dependent and nonlinear — which a
   second-order covariance probe cannot see. So the kill is precise: order does not appear as
   content-order skew in the residual-stream lag-covariance. It is **not** a claim that the model
   has no notion of order. Chasing that further needs a different instrument (attention-level /
   nonlinear), which is outside this repo's scope.

---

## The map — every file, what it is, its status

Status legend: **[V]** verified · **[K]** killed · **[B]** bet · **[LIVE]** runnable instrument.

### `ember/` — the engine
| file | what it is | status |
|---|---|---|
| `cell.py` | the Ember cell: symmetric settle `S`, skew flow `A`, excitable refractory surprise gate, protected rotor latch (soft commit), two readout heads. | **[V]** runs / **[K]** as a GRU-beater |
| `cell_hard.py` | identical cell, rotor commit made a **hard straight-through flip**. The thing exp 03 tests. | **[V]** runs / **[K]** as a latch fix |
| `probe.py` | the `S ⊕ A` diagnostic + shuffle falsification; `harvest_hidden_states` hook for a real transformer. | **[V][LIVE]** |
| `baselines.py` | parameter-matched GRU + the gate-ablated control. | **[V]** |
| `data.py` | the two synthetic tasks: smooth carrier and parity-of-surprises. | **[V]** |
| `optim.py` | minimal Adam over a dict of params. | **[V]** |

### `experiments/`
| file | what it is | status |
|---|---|---|
| `01_probe_sanity.py` | falsifies the premise on synthetic ground truth (shuffle must collapse the arrow). | **[V]** premise holds (5.05×) |
| `02_kill_keep.py` | Ember vs gate-ablation vs GRU on parity; accuracy + gate concentration. | **[V/K]** mixed, reported honestly |
| `03_fix_the_latch.py` | soft vs hard straight-through latch vs GRU; 3-seed gap + rotor-drift diagnostic. | **[K]** discreteness helps but doesn't close it |
| `04_probe_real_model.py` | the probe on GPT-2 across τ = 1–16: shuffle-ratio by layer, content-order vs scramble, middle-peak test. | **[K]** premise does not transfer at any lag |

### `continuous/` — does the economy pay on a stream? (see its own README)
| file | what it is | status |
|---|---|---|
| `efficiency.py` | one operating point on a noisy switching-oscillator. | **[V]** arrow helps / **[K]** gate doesn't concentrate |
| `economy_regime.py` | a noise sweep mapping where event-driven gating would pay. | **[K]** ~1.0× everywhere |
| `continuous.py` | switching-oscillator generator + `from_timeseries()` for real audio/video. | **[V][LIVE]** |

---

## The ledger, consolidated

**Verified in code:**
- the `S ⊕ A` premise **on synthetic ground truth** — `A` carries order; shuffling collapses
  ‖A‖ ~5× (`01`);
- the economy — a refractory surprise-gate keeps the directional flow dark on predictable steps
  and fires on surprise (concentration 3–20× on a task that rewards detecting events) (`02`);
- the gate is load-bearing — ablating it collapses the memory task to chance (`02`);
- discreteness gives a small real bump — the hard flip beats the soft latch on parity (`03`).

**Killed by the builds (the useful negatives):**
- "the surprise-gate concentrates on a smooth next-step task" — **false**: the cheap `S`-half
  already solves smooth prediction. You cannot test an economy on a task `S` solves for free.
- "an unbounded winding read by a linear head stores a discrete count" — **false**: same failure
  as the parent line's phase wheel.
- "the cell beats a recurrent baseline at long-horizon discrete memory" — **false here**:
  GRU ~0.91 vs Ember 0.53–0.71 on 64-step parity.
- "a hard, discrete flip fixes the latch" — **false** (`03`): helps parity a little but *raises*
  drift, **collapses concentration 3.4×→1.0×**, and still loses to the GRU. It routes around the
  latch rather than holding it.
- "the surprise-gate spontaneously concentrates on events during continuous prediction" —
  **false**: ~1.0× concentration across noise levels.
- "the `A`-premise transfers to a real transformer (content-order ‖A‖ peaks mid-stack)" —
  **false at every lag** (`04`): on GPT-2 across τ = 1–16 the mid-stack arrow stays below the
  shuffle floor and content order never beats scramble. The only above-floor structure is
  **positional** (an embedding-layer arrow that grows with lag and is content-independent:
  ordered ≈ scrambled at τ≥8) and a **readout** signal at the output layer. Consistent with the
  framing (the transformer's arrow is a stapled-on positional coordinate); not evidence of an
  emergent content-order flow. Scope: a linear second-order probe can't see attention-routed order.

**Still a bet:**
- that **supervising the gate** toward the model's own normalized prediction-error keeps
  concentration high — the prerequisite exp 03 showed is missing, and the clear next training move;
- that, with a supervised selective gate, a hard straight-through latch then has something to
  latch onto and can hold over a long horizon.

---

## What would actually move this

The probe thread is closed: exp 04 swept the lag and the residual-stream arrow is positional and
edge-localized on GPT-2, full stop. The live work is back on the cell, and exp 03 set the order.

1. **Supervise the gate (the clear next training move).** Add a term pushing the gate toward the
   model's own normalized prediction-error; re-run `03` and `continuous/economy_regime.py` and see
   whether concentration stays high *while* parity improves. Prerequisite for the latch and for any
   real-stream efficiency claim.
2. **Then re-test the hard latch *with* a supervised gate.** Only after (1) does the straight-
   through flip from `cell_hard.py` have a selective signal to fire on; that is the test of whether
   the 0.53–0.71 → 0.91 gap to the GRU can be closed.
3. **Then point it at real audio/video.** `continuous.py:from_timeseries()` frames any real stream
   into the harness — but only after (1), or the gate will sit at a constant.
4. *(Optional, out of scope here)* if the transformer-arrow question is worth chasing past the
   second-order probe, it needs an attention-level / nonlinear instrument — the residual-stream
   lag-covariance has said its piece.

---

## Lineage

Built on the GeometricNeuron / GAIT / PerceptionLab line (`github.com/anttiluode`),
specifically the V13→V14 result that direction must be *generated* by an active element,
the whorl's topological latch, and `the_tensor`'s spend-on-surprise economy. The original
insight and direction are Antti Luode's; this build, its experiments, and this ledger were
written with Claude (Opus 4.8).

*Do not hype. Do not lie. Just show.*
