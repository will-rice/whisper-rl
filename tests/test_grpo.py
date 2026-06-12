"""Tests for the core GRPO math helpers."""

import torch

from whisper_rl.grpo import (
    completion_mask_from_ids,
    group_advantages,
    grpo_loss,
    kl_divergence,
    sequence_log_probs,
)


def test_group_advantages_are_zero_mean_per_group() -> None:
    """Each group of advantages should center to (near) zero mean."""
    rewards = torch.tensor([0.0, 1.0, 2.0, 10.0, 20.0, 30.0])
    advantages = group_advantages(rewards, num_generations=3, eps=0.0)
    grouped = advantages.view(-1, 3)
    assert torch.allclose(grouped.mean(dim=1), torch.zeros(2), atol=1e-6)
    # Within a group, higher reward yields higher advantage.
    assert (grouped[:, 2] > grouped[:, 0]).all()


def test_group_advantages_constant_group_is_finite() -> None:
    """A zero-variance group should not produce NaNs thanks to eps."""
    rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
    advantages = group_advantages(rewards, num_generations=2, eps=1e-4)
    assert torch.isfinite(advantages).all()
    assert torch.allclose(advantages, torch.zeros_like(advantages))


def test_sequence_log_probs_matches_manual_gather() -> None:
    """Gathered log-probs should match a manual log-softmax gather."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))
    log_probs = sequence_log_probs(logits, targets)
    expected = (
        torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    )
    assert torch.allclose(log_probs, expected)


def test_kl_divergence_is_non_negative_and_zero_when_equal() -> None:
    """The KL estimator is non-negative and zero for identical log-probs."""
    policy = torch.randn(4, 6)
    assert torch.allclose(kl_divergence(policy, policy), torch.zeros(4, 6), atol=1e-6)
    other = policy + 0.5
    assert (kl_divergence(policy, other) >= 0).all()


def test_completion_mask_stops_after_first_eos() -> None:
    """Masking keeps tokens up to and including the first EOS."""
    ids = torch.tensor([[3, 4, 9, 9], [9, 1, 2, 3]])
    mask = completion_mask_from_ids(ids, eos_token_id=9)
    assert torch.equal(mask, torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]]))


def test_completion_mask_no_eos_keeps_all() -> None:
    """A row with no EOS keeps every token."""
    ids = torch.tensor([[1, 2, 3]])
    mask = completion_mask_from_ids(ids, eos_token_id=9)
    assert torch.equal(mask, torch.ones(1, 3, dtype=ids.dtype))


def test_grpo_loss_reduces_to_negative_advantage_on_policy() -> None:
    """With ratio=1 and KL=0 the loss equals the negative mean advantage."""
    log_probs = torch.randn(2, 4, requires_grad=True)
    advantages = torch.tensor([1.0, -2.0])
    mask = torch.ones(2, 4)
    loss, mean_kl = grpo_loss(
        log_probs,
        log_probs.detach(),
        log_probs.detach(),
        advantages,
        mask,
        clip_eps=0.2,
        kl_beta=0.04,
    )
    assert torch.allclose(loss, -advantages.mean())
    assert torch.allclose(mean_kl, torch.zeros(()))
    loss.backward()
    assert log_probs.grad is not None


def test_grpo_loss_respects_mask() -> None:
    """Masked-out tokens must not contribute to the loss."""
    log_probs = torch.zeros(1, 4)
    advantages = torch.tensor([2.0])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss, _ = grpo_loss(log_probs, log_probs, log_probs, advantages, mask, kl_beta=0.0)
    # Only two unmasked tokens, each contributing -advantage.
    assert torch.allclose(loss, torch.tensor(-2.0))
