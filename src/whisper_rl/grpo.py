"""Core GRPO math: group advantages, token log-probs, and the loss.

These helpers are intentionally free of any model or dataset dependencies so
they can be unit tested in isolation. They implement Group Relative Policy
Optimization (GRPO, Shao et al. 2024) as used to finetune a sequence model:

* sample a group of ``G`` completions per prompt,
* score each completion with a reward,
* center and scale rewards within each group to form advantages,
* optimize a clipped policy-gradient objective with a per-token KL penalty
  against a frozen reference policy.
"""

import torch
import torch.nn.functional as F  # noqa: N812


def group_advantages(
    rewards: torch.Tensor, num_generations: int, eps: float = 1e-4
) -> torch.Tensor:
    """Normalize rewards within each group to produce advantages.

    Args:
        rewards: Flat tensor of shape ``(batch * num_generations,)`` ordered so
            the ``num_generations`` completions of a prompt are contiguous.
        num_generations: Number of completions sampled per prompt (``G``).
        eps: Constant added to the per-group standard deviation for stability.

    Returns:
        Advantages of the same shape as ``rewards``, with each group centered
        to zero mean and scaled by its standard deviation.
    """
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


def sequence_log_probs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Gather per-token log-probabilities of ``targets`` under ``logits``.

    Args:
        logits: Decoder logits of shape ``(batch, seq_len, vocab)`` where
            position ``t`` scores the distribution over ``targets[:, t]``.
        targets: Token ids of shape ``(batch, seq_len)``.

    Returns:
        Per-token log-probabilities of shape ``(batch, seq_len)``.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def sft_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked, token-averaged supervised cross-entropy (teacher forcing).

    Unlike the GRPO term, which only reweights the model's *own* samples, this
    is a direct supervised signal toward the reference tokens, so it can teach
    sequences the policy would never sample on its own.

    Args:
        logits: Decoder logits of shape ``(batch, seq_len, vocab)`` where
            position ``t`` scores the distribution over ``targets[:, t]``.
        targets: Reference token ids of shape ``(batch, seq_len)``.
        mask: ``1`` on the reference tokens to supervise, ``0`` on prompt and
            padding positions, shape ``(batch, seq_len)``.

    Returns:
        Scalar negative log-likelihood averaged over the masked tokens.
    """
    log_probs = sequence_log_probs(logits, targets)
    mask = mask.to(log_probs.dtype)
    token_count = mask.sum().clamp(min=1.0)
    return -(log_probs * mask).sum() / token_count


def sft_weight_at(step: int, start: float, final: float, anneal_steps: int) -> float:
    """Linearly annealed weight of the SFT term at a training step.

    The supervised term teaches languages the policy never samples correctly, so
    it is most useful early (to bootstrap them) but over-corrects a strong base
    model late. Decay it linearly from ``start`` at step 0 to ``final`` at
    ``anneal_steps``, then hold ``final``.

    Args:
        step: Current global optimizer step.
        start: SFT weight at step 0.
        final: SFT weight at and after ``anneal_steps``.
        anneal_steps: Steps over which to decay; ``<= 0`` uses ``final`` at once.

    Returns:
        The SFT weight to apply at ``step``.
    """
    if anneal_steps <= 0 or step >= anneal_steps:
        return final
    return start + (final - start) * (step / anneal_steps)


def kl_divergence(
    policy_log_probs: torch.Tensor, ref_log_probs: torch.Tensor
) -> torch.Tensor:
    """Per-token unbiased KL estimate of policy from reference.

    Uses the low-variance, non-negative estimator from Schulman's blog and the
    GRPO paper: ``exp(ref - pol) - (ref - pol) - 1``.

    Args:
        policy_log_probs: Log-probs under the current policy.
        ref_log_probs: Log-probs under the frozen reference policy.

    Returns:
        Per-token KL estimate, elementwise non-negative.
    """
    diff = ref_log_probs - policy_log_probs
    return torch.exp(diff) - diff - 1.0


def grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the masked, token-averaged GRPO loss.

    Args:
        policy_log_probs: Current-policy per-token log-probs,
            shape ``(batch, seq_len)``.
        old_log_probs: Sampling-policy per-token log-probs (detached),
            shape ``(batch, seq_len)``. For a single update per rollout these
            equal ``policy_log_probs.detach()`` and the ratio is ``1``.
        ref_log_probs: Reference-policy per-token log-probs (detached),
            shape ``(batch, seq_len)``.
        advantages: Per-sequence advantages of shape ``(batch,)``.
        completion_mask: ``1`` for real completion tokens, ``0`` for padding /
            prompt tokens, shape ``(batch, seq_len)``.
        clip_eps: PPO-style clipping epsilon.
        kl_beta: Weight of the KL penalty term.

    Returns:
        A tuple ``(loss, mean_kl)`` where ``loss`` is the scalar objective to
        minimize and ``mean_kl`` is the mask-averaged KL (for logging).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    advantages = advantages.unsqueeze(1)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_term = torch.min(unclipped, clipped)

    kl = kl_divergence(policy_log_probs, ref_log_probs)
    per_token_loss = -(policy_term - kl_beta * kl)

    mask = completion_mask.to(per_token_loss.dtype)
    token_count = mask.sum().clamp(min=1.0)
    loss = (per_token_loss * mask).sum() / token_count
    mean_kl = (kl * mask).sum() / token_count
    return loss, mean_kl


def completion_mask_from_ids(
    completion_ids: torch.Tensor, eos_token_id: int
) -> torch.Tensor:
    """Build a mask that keeps completion tokens up to and including EOS.

    Tokens generated after the first end-of-sequence token are padding and are
    masked out so they do not contribute to the loss.

    Args:
        completion_ids: Generated token ids of shape ``(batch, seq_len)``,
            excluding the forced decoder prompt.
        eos_token_id: The end-of-sequence token id.

    Returns:
        A float mask of shape ``(batch, seq_len)``.
    """
    is_eos = completion_ids == eos_token_id
    seq_len = completion_ids.size(1)
    # Index of the first EOS in each row, or seq_len if none is present.
    first_eos = torch.where(
        is_eos.any(dim=1),
        is_eos.float().argmax(dim=1),
        torch.full((completion_ids.size(0),), seq_len, device=completion_ids.device),
    )
    positions = torch.arange(seq_len, device=completion_ids.device).unsqueeze(0)
    return (positions <= first_eos.unsqueeze(1)).to(completion_ids.dtype)
