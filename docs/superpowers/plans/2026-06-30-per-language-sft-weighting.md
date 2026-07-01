# Adaptive per-language SFT weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Weight each clip's SFT term by a scaled ramp of its language's smoothed validation CER, so strong languages are protected while floored languages keep full teaching.

**Architecture:** Three pure helpers in `grpo.py` (per-clip weighted SFT loss, the CER→weight map lookup, and the EMA update) plus wiring in `lightning_module.py` (hold a per-language CER map, EMA-update it each validation, build per-clip weights in the training step). A `sft_adaptive` config flag gates the new path; the existing anneal path stays byte-identical.

**Tech Stack:** PyTorch, PyTorch Lightning, pytest, `uv`.

## Global Constraints

- Weight function: `weight = clamp(cer / cer_ref, floor, cap)` with `cer_ref = 0.4`, `floor = sft_weight_final = 0.1`, `cap = sft_weight = 1.0`.
- A language not yet in the CER map gets weight `0` (no SFT until measured).
- EMA-update per-language CER each validation: `sft_cer[lang] = ema*old + (1-ema)*new`, seeded exactly with the first observed CER; `ema = sft_cer_ema = 0.7`.
- Loss aggregation: `weighted_sft_loss = mean_i(weight_i * per_clip_nll_i)`; `per_clip_nll_i = -(log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)`.
- The non-adaptive (anneal) path is unchanged: `loss = policy_loss + sft_weight_at(...) * sft_loss(...)`.
- Do NOT update the CER map during `trainer.sanity_checking` (keeps the zero-init warmup intact).
- New config fields: `sft_adaptive: bool = False`, `sft_cer_ref: float = 0.4`, `sft_cer_ema: float = 0.7`. Reuse `sft_weight` (cap) and `sft_weight_final` (floor).
- Run pre-commit for every commit: `uv run pre-commit run -a` (ruff, ty, pytest). If it fails with `No module named 'whisper_rl'`, run `uv pip install -e . --quiet` first (editable-install quirk) and rerun.

---

### Task 1: `weighted_sft_loss` in grpo.py

**Files:**

- Modify: `src/whisper_rl/grpo.py`
- Test: `tests/test_grpo.py`

**Interfaces:**

- Consumes: `sequence_log_probs` (already in `grpo.py`).
- Produces: `weighted_sft_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor` — scalar; `mean_i(weights_i * per_clip_nll_i)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_grpo.py` (and add `weighted_sft_loss` to the existing `from whisper_rl.grpo import (...)`):

```python
def test_weighted_sft_loss_uniform_weights_is_per_clip_mean() -> None:
    """Uniform weights reduce to the mean of per-clip token-averaged NLL."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    lp = torch.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    per_clip = -(lp * mask).sum(dim=1) / mask.sum(dim=1)
    got = weighted_sft_loss(logits, targets, mask, torch.tensor([1.0, 1.0]))
    assert torch.allclose(got, per_clip.mean())


def test_weighted_sft_loss_zero_weight_drops_clip() -> None:
    """A zero-weight clip contributes nothing but still counts in the mean."""
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))
    mask = torch.ones(2, 3)
    lp = torch.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    per_clip = -(lp * mask).sum(dim=1) / mask.sum(dim=1)
    got = weighted_sft_loss(logits, targets, mask, torch.tensor([1.0, 0.0]))
    assert torch.allclose(got, per_clip[0] / 2)


def test_weighted_sft_loss_all_zero_is_zero() -> None:
    """An all-zero weight vector yields zero SFT (pure-GRPO warmup)."""
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))
    got = weighted_sft_loss(logits, targets, torch.ones(2, 3), torch.zeros(2))
    assert got == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_grpo.py -k weighted_sft_loss -q`
Expected: FAIL — `ImportError: cannot import name 'weighted_sft_loss'`.

- [ ] **Step 3: Implement**

In `src/whisper_rl/grpo.py`, add after `sft_loss`:

```python
def weighted_sft_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-clip-weighted masked SFT loss.

    Each clip's token-averaged negative log-likelihood is scaled by its weight
    and averaged over the batch, so a batch of low-weight (protected) clips
    contributes proportionally little SFT gradient.

    Args:
        logits: Decoder logits of shape ``(batch, seq_len, vocab)``.
        targets: Reference token ids of shape ``(batch, seq_len)``.
        mask: ``1`` on supervised reference tokens, ``0`` elsewhere,
            shape ``(batch, seq_len)``.
        weights: Per-clip SFT weight of shape ``(batch,)``.

    Returns:
        Scalar ``mean_i(weights_i * nll_i)``.
    """
    log_probs = sequence_log_probs(logits, targets)
    mask = mask.to(log_probs.dtype)
    per_clip_nll = -(log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return (weights.to(per_clip_nll.dtype) * per_clip_nll).mean()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_grpo.py -k weighted_sft_loss -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/whisper_rl/grpo.py tests/test_grpo.py
git commit -m "Add per-clip weighted SFT loss"
```

---

### Task 2: `sft_weights_for` in grpo.py

**Files:**

- Modify: `src/whisper_rl/grpo.py`
- Test: `tests/test_grpo.py`

**Interfaces:**

- Produces: `sft_weights_for(languages: list[str], cer_map: dict[str, float], cer_ref: float, floor: float, cap: float) -> list[float]` — one weight per language; measured language → `clamp(cer/cer_ref, floor, cap)`, unmeasured → `0.0`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_grpo.py` (add `sft_weights_for` to the import):

```python
def test_sft_weights_for_ramps_and_clamps() -> None:
    """Measured languages map to clamp(cer/cer_ref, floor, cap)."""
    cer_map = {"hi": 0.8, "de": 0.05, "mr": 0.2}
    w = sft_weights_for(["hi", "de", "mr"], cer_map, 0.4, 0.1, 1.0)
    assert w[0] == 1.0            # 0.8/0.4 = 2.0 -> cap
    assert w[1] == 0.1            # 0.05/0.4 = 0.125 -> floor
    assert abs(w[2] - 0.5) < 1e-9  # 0.2/0.4 = 0.5


def test_sft_weights_for_unmeasured_language_is_zero() -> None:
    """A language absent from the map gets no SFT yet."""
    assert sft_weights_for(["ja"], {}, 0.4, 0.1, 1.0) == [0.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_grpo.py -k sft_weights_for -q`
Expected: FAIL — `ImportError: cannot import name 'sft_weights_for'`.

- [ ] **Step 3: Implement**

In `src/whisper_rl/grpo.py`, add after `weighted_sft_loss`:

```python
def sft_weights_for(
    languages: list[str],
    cer_map: dict[str, float],
    cer_ref: float,
    floor: float,
    cap: float,
) -> list[float]:
    """Per-clip SFT weights from smoothed per-language CER.

    A language with a measured CER gets ``clamp(cer / cer_ref, floor, cap)`` — a
    scaled ramp that is full at ``cer_ref`` error and falls to the floor as the
    language improves. A language not yet measured gets ``0`` (no SFT until its
    error is known), which is below the floor and so distinguishable from a
    measured-and-protected language.

    Args:
        languages: Per-clip language codes (unrepeated batch order).
        cer_map: Smoothed CER per language.
        cer_ref: CER at or above which a language gets the full ``cap``.
        floor: Minimum weight for a measured language.
        cap: Maximum weight.

    Returns:
        One weight per entry in ``languages``.
    """
    return [
        min(cap, max(floor, cer_map[lang] / cer_ref)) if lang in cer_map else 0.0
        for lang in languages
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_grpo.py -k sft_weights_for -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/whisper_rl/grpo.py tests/test_grpo.py
git commit -m "Add per-language SFT weight ramp with zero cold-start"
```

---

### Task 3: `ema_update` in grpo.py

**Files:**

- Modify: `src/whisper_rl/grpo.py`
- Test: `tests/test_grpo.py`

**Interfaces:**

- Produces: `ema_update(cer_map: dict[str, float], new_cer: dict[str, float], ema: float) -> None` — mutates `cer_map` in place; seeds unseen languages with the observed CER, EMA-blends seen ones.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_grpo.py` (add `ema_update` to the import):

```python
def test_ema_update_seeds_on_first_sight() -> None:
    """The first observed CER is stored exactly (no ramp from zero)."""
    cer_map: dict[str, float] = {}
    ema_update(cer_map, {"hi": 0.6}, 0.7)
    assert cer_map["hi"] == 0.6


def test_ema_update_blends_existing() -> None:
    """A subsequent CER moves the value by (1 - ema) toward the new value."""
    cer_map = {"hi": 0.6}
    ema_update(cer_map, {"hi": 0.4}, 0.7)
    assert abs(cer_map["hi"] - (0.7 * 0.6 + 0.3 * 0.4)) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_grpo.py -k ema_update -q`
Expected: FAIL — `ImportError: cannot import name 'ema_update'`.

- [ ] **Step 3: Implement**

In `src/whisper_rl/grpo.py`, add after `sft_weights_for`:

```python
def ema_update(cer_map: dict[str, float], new_cer: dict[str, float], ema: float) -> None:
    """Exponential-moving-average update of the per-language CER map, in place.

    A language seen for the first time is stored exactly (so its SFT weight is
    correct from its first validation); a language already present is blended
    ``ema * old + (1 - ema) * new`` to damp the per-validation noise of a small
    per-language eval slice.

    Args:
        cer_map: The map to update in place.
        new_cer: Freshly measured CER per language (exclude the ``overall`` key).
        ema: Weight on the existing value in ``[0, 1)``.
    """
    for lang, cer in new_cer.items():
        cer_map[lang] = ema * cer_map[lang] + (1 - ema) * cer if lang in cer_map else cer
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_grpo.py -k ema_update -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/whisper_rl/grpo.py tests/test_grpo.py
git commit -m "Add EMA update for per-language CER map"
```

---

### Task 4: Config fields and lightning_module wiring

**Files:**

- Modify: `src/whisper_rl/config.py`
- Modify: `src/whisper_rl/lightning_module.py`

**Interfaces:**

- Consumes: `weighted_sft_loss`, `sft_weights_for`, `ema_update` (Tasks 1–3).
- Produces: an `sft_adaptive` training path; `self.sft_cer: dict[str, float]` on the module.

- [ ] **Step 1: Add the config fields**

In `src/whisper_rl/config.py`, immediately after the `sft_anneal_end` line, add:

```python
    # Per-language adaptive SFT: when true, each clip's SFT term is weighted by
    # clamp(cer / sft_cer_ref, sft_weight_final, sft_weight) using its language's
    # EMA-smoothed validation CER (unmeasured languages get 0). Replaces the
    # step-based anneal above.
    sft_adaptive: bool = False
    sft_cer_ref: float = 0.4
    sft_cer_ema: float = 0.7
```

- [ ] **Step 2: Import the helpers**

In `src/whisper_rl/lightning_module.py`, extend the `from whisper_rl.grpo import (...)` block to include `ema_update`, `sft_weights_for`, and `weighted_sft_loss` (keep the list alphabetized: `..., sft_loss, sft_weight_at, sft_weights_for, weighted_sft_loss` and `ema_update` near the top):

```python
from whisper_rl.grpo import (
    completion_mask_from_ids,
    ema_update,
    group_advantages,
    grpo_loss,
    sequence_log_probs,
    sft_loss,
    sft_weight_at,
    sft_weights_for,
    weighted_sft_loss,
)
```

- [ ] **Step 3: Initialize the CER map**

In `__init__`, right after `self.val_metric = LanguageErrorRate()`, add:

```python
        self.sft_cer: dict[str, float] = {}
```

- [ ] **Step 4: Add an optional `weights` arg to `_sft_loss`**

Change the `_sft_loss` signature to accept weights, and change its return. Signature:

```python
    def _sft_loss(
        self,
        input_features: torch.Tensor,
        prompt: torch.Tensor,
        references: list[str],
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
```

Replace the final line `return sft_loss(logits, sequences[:, 1:], mask)` with:

```python
        if weights is not None:
            return weighted_sft_loss(logits, sequences[:, 1:], mask, weights)
        return sft_loss(logits, sequences[:, 1:], mask)
```

- [ ] **Step 5: Branch the training-step loss**

In `training_step`, replace the current block:

```python
        supervised = self._sft_loss(batch.input_features, base_prompt, batch.references)
        sft_weight = sft_weight_at(
            self.global_step,
            self.config.sft_weight,
            self.config.sft_weight_final,
            self.config.sft_anneal_start,
            self.config.sft_anneal_end,
        )
        loss = policy_loss + sft_weight * supervised
```

with:

```python
        if self.config.sft_adaptive:
            weights = torch.tensor(
                sft_weights_for(
                    batch.languages,
                    self.sft_cer,
                    self.config.sft_cer_ref,
                    self.config.sft_weight_final,
                    self.config.sft_weight,
                ),
                device=self.device,
            )
            supervised = self._sft_loss(
                batch.input_features, base_prompt, batch.references, weights=weights
            )
            sft_weight = weights.mean()
            loss = policy_loss + supervised
        else:
            supervised = self._sft_loss(
                batch.input_features, base_prompt, batch.references
            )
            sft_weight = sft_weight_at(
                self.global_step,
                self.config.sft_weight,
                self.config.sft_weight_final,
                self.config.sft_anneal_start,
                self.config.sft_anneal_end,
            )
            loss = policy_loss + sft_weight * supervised
```

The existing `self.log("train/sft_weight", sft_weight, batch_size=batch_size)` line is unchanged and logs the mean weight in adaptive mode.

- [ ] **Step 6: EMA-update the map each validation**

In `on_validation_epoch_end`, right after `results = self.val_metric.compute()`, add (the sanity guard keeps the zero-init warmup intact):

```python
        if self.config.sft_adaptive and not self.trainer.sanity_checking:
            ema_update(
                self.sft_cer,
                {
                    lang: cer
                    for lang, cer in results["cer"].items()
                    if lang != "overall"
                },
                self.config.sft_cer_ema,
            )
```

- [ ] **Step 7: Run pre-commit (ruff, ty, full pytest)**

Run: `uv run pre-commit run -a`
Expected: all hooks Pass. (If pytest reports `No module named 'whisper_rl'`, run `uv pip install -e . --quiet` then rerun.) `ty` verifies the `_sft_loss` signature and the new branch typecheck.

- [ ] **Step 8: Commit**

```bash
git add src/whisper_rl/config.py src/whisper_rl/lightning_module.py
git commit -m "Wire adaptive per-language SFT weighting into training"
```

---

### Task 5: Smoke test, PR, and launch the experiment

**Files:**

- None (operational; runs on `green`).

**Interfaces:**

- Consumes: the merged feature.
- Produces: the `cv22sftadaptive` run.

- [ ] **Step 1: PR and merge**

```bash
git push -u origin per-language-sft-weighting
gh pr create --title "Adaptive per-language SFT weighting" --body "Implements docs/superpowers/specs/2026-06-30-per-language-sft-weighting-design.md"
```

Wait for CI (3.10/3.11/3.12) to pass, then `gh pr merge <n> --merge --delete-branch`.

- [ ] **Step 2: Deploy to green and set the recipe + sft_adaptive=True**

Only launch once the Ada is free (the `cv22sftanneal2` baseline finishes its ~40k read). Then on `green`:

```bash
ssh green 'cd ~/projects/whisper-rl && git checkout -- src/whisper_rl/config.py && git pull --ff-only origin main && \
sed -i "s|dataset_name: str = \"fixie-ai/common_voice_17_0\"|dataset_name: str = \"/data/cv22_index\"|" src/whisper_rl/config.py && \
sed -i "s/^    batch_size: int = 8/    batch_size: int = 16/" src/whisper_rl/config.py && \
sed -i "s/^    temperature: float = 1.0/    temperature: float = 0.7/" src/whisper_rl/config.py && \
sed -i "s/^    max_steps: int = 500/    max_steps: int = 1000000/" src/whisper_rl/config.py && \
sed -i "s/^    learning_rate: float = 1e-6/    learning_rate: float = 1e-5/" src/whisper_rl/config.py && \
sed -i "s/^    sft_adaptive: bool = False/    sft_adaptive: bool = True/" src/whisper_rl/config.py && \
grep -nE "dataset_name|batch_size|temperature|max_steps|learning_rate|sft_adaptive|sft_cer_ref|sft_cer_ema|sft_weight|sft_weight_final" src/whisper_rl/config.py'
```

Expected: recipe values patched and `sft_adaptive = True`.

- [ ] **Step 3: Fast-dev-run smoke (no W&B, no GPU commitment)**

```bash
ssh green 'cd ~/projects/whisper-rl && export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$HOME/micromamba/envs/ffmpeg7/lib && ~/.local/bin/uv run train --run_suffix smoke --fast_dev_run --no_wandb 2>&1 | tail -20'
```

Expected: one train + val step complete with no traceback (confirms the adaptive branch and the EMA update run end to end).

- [ ] **Step 4: Launch the run**

Write `~/cv22_sftadaptive_supervisor.sh` on green with this content:

```bash
#!/usr/bin/env bash
cd ~/projects/whisper-rl
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=$HOME/micromamba/envs/ffmpeg7/lib
export WANDB_RUN_ID=cv22sftadaptive WANDB_RESUME=allow
SUFFIX=cv22-tiny-sftadaptive
for i in $(seq 1 100); do
  CK=$(ls -t logs/*$SUFFIX/last.ckpt 2>/dev/null | head -1)
  ARG=""; [ -n "$CK" ] && ARG="--checkpoint_path $CK"
  echo "=== attempt $i $(date -u) ckpt=${CK:-none} ==="
  $HOME/.local/bin/uv run train --run_suffix $SUFFIX $ARG
  code=$?; echo "=== exited $code (attempt $i) ==="
  [ $code -eq 0 ] && break
  sleep 30
done
```

Then `chmod +x` it and launch:

```bash
ssh green 'chmod +x ~/cv22_sftadaptive_supervisor.sh && tmux new-session -d -s sftadaptive "bash ~/cv22_sftadaptive_supervisor.sh 2>&1 | tee -a /data/cv22_sftadaptive.log"; sleep 5; tmux has-session -t sftadaptive && echo LAUNCHED'
```

- [ ] **Step 5: Verify config and the zero-init warmup**

After the first validation (~250 steps), check `~/cv22_monitor.py cv22sftadaptive` and confirm in the W&B run config that `sft_adaptive=True`, `sft_cer_ref=0.4`, `sft_cer_ema=0.7`. Confirm `train/sft_weight` starts near 0 (pure-GRPO warmup) and rises after the first validation. Compare overall `val/cer` at 20k/40k against `cv22sftanneal2` and the dead run, and watch en/de/pt (should not degrade) and te/ml/bn (should keep improving).
