# Adaptive per-language SFT weighting — design

**Date:** 2026-06-30
**Goal:** Push overall CV22 validation error below the hybrid GRPO+SFT plateau by
weighting each clip's SFT term by how much _its language_ still needs teaching —
protecting the languages the base model is already strong at while keeping full
teaching on the ones that are still floored.

## Motivation

On full CV22, a static (and even a globally-annealed) `sft_weight` cannot serve
all languages at once. The dead run's per-language curves show:

- The languages Whisper-tiny is **strongest** at degrade under heavy SFT, and
  only late: en 0.17 → 0.44, de 0.18 → 0.39, pt 0.28 → 0.39 (all turning after
  ~8k). SFT toward CV references hurts most exactly where the base is best.
- The **floored** languages need full SFT and keep improving through ~8k (hi, ml,
  bn, te), and several plateau high (te ~0.61, ml ~0.47, bn ~0.40) — they would
  benefit from SFT _past_ where a global schedule backs it off.

A single global `sft_weight(step)` is one clock for languages that want opposite
things at the same step. Weighting per language dissolves the conflict: strong
languages get near-floor SFT from the start; still-floored languages keep full
SFT until they themselves improve. It is per-language self-annealing.

The global hold-then-decay anneal run (`cv22sftanneal2`) is the baseline this is
measured against.

## Weight function

For a language with current (smoothed) validation CER `cer`:

```
weight = clamp(cer / c_ref, floor, cap)
```

with `c_ref = 0.4`, `floor = 0.1`, `cap = 1.0`. Any language at or above 0.4 CER
gets full SFT; the weight ramps linearly to the floor as a language improves.
CER (not WER) is used so the CJK whitespace-WER artifact does not wrongly pin
ja/zh at full SFT.

`cap` reuses the existing `sft_weight` (1.0) and `floor` reuses
`sft_weight_final` (0.1) — the same "max / min SFT weight" knobs as the anneal
mechanism, no redundant config.

## Weight map (validation → training)

The module holds `self.sft_cer: dict[str, float]`, the per-language smoothed CER.

- In `on_validation_epoch_end`, after `results = self.val_metric.compute()`,
  EMA-update from `results["cer"]` (excluding the `"overall"` key):
  `sft_cer[lang] = ema * sft_cer[lang] + (1 - ema) * cer`, seeded with `cer` on
  first sight. `ema = sft_cer_ema = 0.7`.
- The per-clip weight is computed live in `training_step` from this map, so it
  updates every validation and needs only dict lookups per step.

`LanguageErrorRate.compute()["cer"]` is already produced every validation and is
keyed the same way as `batch.languages`, so no new metric plumbing is needed.

## Cold start

A language with **no** validation CER yet gets weight `0` — no SFT until its
error proves it needs teaching. So:

- Before the first validation (steps 0 to ~250), every clip's weight is `0`: the
  run is pure GRPO for that warmup, and the strong languages take no SFT hit at
  the very start (exactly where en degrades fastest).
- At the first validation the CER map is **seeded with the observed CER** (not
  ramped from zero via the EMA), so each language's weight jumps straight to its
  correct ramp value — floored languages are not starved waiting for an EMA to
  climb. Subsequent validations EMA-update.
- The fixed 256-clip eval covers every language (the dead run reported all ~77),
  so after the first validation no language stays at `0`; a measured language's
  weight is bounded below by the ramp floor (`0.1`). A weight of `0` therefore
  means only "not yet measured", never "measured and protected".

## Loss plumbing

New pure helper in `grpo.py`:

```
weighted_sft_loss(logits, targets, mask, weights) -> scalar
    per_clip_nll_i = -(log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return mean_i(weights_i * per_clip_nll_i)
```

Aggregating as `mean_i(w_i · nll_i)` (not a weight-normalized mean) means a batch
of protected languages contributes proportionally little SFT gradient — which is
the point.

`_sft_loss` gains an optional `weights: torch.Tensor | None = None` argument: it
builds the logits/targets/mask exactly as today, then returns
`weighted_sft_loss(logits, targets, mask, weights)` when weights are given and
`sft_loss(logits, targets, mask)` when they are not. `training_step` branches on
`config.sft_adaptive`:

- **adaptive:** build a per-clip weight tensor from `batch.languages` and the CER
  map, call `self._sft_loss(..., weights=w)`, add it to the loss directly (the
  weights already carry the magnitude), and log `train/sft_weight` as the batch
  mean weight.
- **not adaptive (default):** unchanged — `sft_weight_at(step, ...) *
self._sft_loss(...)` with no weights. The anneal baseline path stays
  byte-identical, so `cv22sftanneal2` remains reproducible.

## Config (new fields)

```
sft_adaptive: bool = False    # off by default; the experiment run sets True
sft_cer_ref: float = 0.4      # CER at/above which a language gets full SFT
sft_cer_ema: float = 0.7      # EMA coefficient on per-language CER
```

Reuses `sft_weight` (cap) and `sft_weight_final` (floor). The anneal fields
(`sft_anneal_start`, `sft_anneal_end`) are simply unused when `sft_adaptive` is
true.

## Experiment

Fresh `openai/whisper-tiny`, full CV22 (`/data/cv22_index`), the exact dead-run
recipe (bs16, temp 0.7, lr 1e-5, kl_beta 0.04, eval 256, 1M `max_steps`,
blended reward), with `sft_adaptive = True`. On the Ada, sequentially after the
`cv22sftanneal2` baseline (only one tiny bs16 fits the Ada; the 3090 is taken by
the user's rogii-2026 project).

- **Baselines:** dead run `cv22tinysft3` (static SFT, val/cer 0.284 plateau) and
  `cv22sftanneal2` (hold-6k→floor-12k global anneal).
- **Primary metric:** overall `val/cer`. Read at ~20k and ~40k.
- **Diagnostic checks:** en/de/pt should _not_ degrade (protected from step 0);
  te/ml/bn should keep improving past 12k (they retain SFT while still high-CER).

## Testing

- `weighted_sft_loss`: uniform weights reduce to the plain per-clip mean; a
  zero-weight clip contributes nothing; a higher-weight clip contributes
  proportionally more; an all-zero weight vector yields `0` (pure-GRPO warmup).
- CER → weight ramp: `clamp` bounds at floor and cap; an **unmeasured** language
  yields weight `0`; a measured language yields the ramp value, never below the
  floor.
- EMA update: the first observed CER seeds the value exactly (weight is correct
  from the first validation); subsequent updates move it by `(1 - ema)` toward
  the new CER.

## Risks / open points

- **Weights ride on a ~3-clip-per-language eval**, smoothed by EMA. If weights
  still look jittery in `train/sft_weight` / per-language behavior, raising
  `max_eval_samples` is the follow-up lever (deferred to preserve eval
  comparability with the baseline).
- **`c_ref = 0.4` is a guess.** It sets which languages count as "needs full
  teaching." If most languages sit below it, SFT may be too weak overall; the
  value is a single, easily-swept knob.
- **Batch composition** determines the overall SFT magnitude each step; with
  interleaved multilingual batches the mean weight is roughly stable, but a batch
  that happens to be mostly strong languages will have little SFT that step (by
  design).
