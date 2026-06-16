"""Lightning module implementing GRPO finetuning of Whisper."""

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from transformers import WhisperProcessor, get_cosine_schedule_with_warmup

from whisper_rl.config import Config
from whisper_rl.datasets import Batch
from whisper_rl.grpo import (
    completion_mask_from_ids,
    group_advantages,
    grpo_loss,
    sequence_log_probs,
)
from whisper_rl.metrics import LanguageErrorRate
from whisper_rl.modeling import (
    build_policy,
    build_processor,
    build_reference,
    decoder_prompt,
    repeat_features,
)
from whisper_rl.rewards import combined_reward


class WhisperGRPOModule(LightningModule):
    """Finetune Whisper with Group Relative Policy Optimization on WER.

    For each audio clip a group of completions is sampled from the current
    policy, conditioned on the clip's known language (pinned from its Common
    Voice locale), scored by negated WER against the reference transcript, and
    turned into group-relative advantages. The policy is updated with a clipped
    policy-gradient objective regularized by a KL penalty toward a frozen copy
    of the initial model. Validation reports word error rate overall and broken
    down per language.
    """

    def __init__(
        self, config: Config, processor: WhisperProcessor | None = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config
        self.processor = processor if processor is not None else build_processor(config)
        self.policy = build_policy(config)
        self.reference = build_reference(config)
        eos = self.policy.config.eos_token_id
        self.eos_token_id = int(eos)  # ty: ignore[invalid-argument-type]
        self.val_metric = LanguageErrorRate()

    def _generate(
        self, input_features: torch.Tensor, prompt: torch.Tensor, sample: bool
    ) -> torch.Tensor:
        """Generate transcriptions conditioned on a fixed decoder ``prompt``.

        The prompt pins each clip's known language (see
        :func:`whisper_rl.modeling.decoder_prompt`); passing it as
        ``decoder_input_ids`` forces that language instead of letting Whisper
        auto-detect (which mislabels lower-resource clips). transformers strips
        the prompt from the output, so the returned ids are completion tokens.

        Args:
            input_features: Features of shape ``(batch, n_mels, frames)``.
            prompt: Decoder prompt ids of shape ``(batch, prompt_len)``.
            sample: Whether to sample (training rollouts) or decode greedily
                (validation).

        Returns:
            Generated completion token ids, without the decoder prompt.
        """
        with torch.no_grad():
            if sample:
                sequences = self.policy.generate(  # ty: ignore[missing-argument]
                    input_features=input_features,
                    decoder_input_ids=prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k if self.config.top_k > 0 else None,
                    num_return_sequences=1,
                )
            else:
                sequences = self.policy.generate(  # ty: ignore[missing-argument]
                    input_features=input_features,
                    decoder_input_ids=prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
        return sequences  # ty: ignore[invalid-return-type]

    def _completion_log_probs(
        self,
        model: torch.nn.Module,
        input_features: torch.Tensor,
        prompt: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token log-probs of ``completion_ids`` conditioned on ``prompt``.

        Args:
            model: The policy or reference Whisper model.
            input_features: Encoder features matching the sequences.
            prompt: Decoder prompt token ids of shape ``(batch, prompt_len)``.
            completion_ids: Generated completion ids of shape
                ``(batch, completion_len)``.

        Returns:
            Per-token log-probs of shape ``(batch, completion_len)``.
        """
        sequences = torch.cat([prompt, completion_ids], dim=1)
        decoder_input_ids = sequences[:, :-1]
        targets = sequences[:, 1:]
        logits = model(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        ).logits
        log_probs = sequence_log_probs(logits, targets)
        return log_probs[:, prompt.size(1) - 1 :]

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Run one GRPO update on a batch of audio clips."""
        num_gen = self.config.num_generations
        features = repeat_features(batch.input_features, num_gen)
        references = [ref for ref in batch.references for _ in range(num_gen)]

        # Pin each clip's known language, then repeat to match the sampled group.
        prompt = decoder_prompt(
            self.policy, batch.input_features, self.config.task, batch.languages
        )
        prompt = prompt.repeat_interleave(num_gen, dim=0)
        completion_ids = self._generate(features, prompt, sample=True)
        hypotheses = self.processor.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        rewards = torch.tensor(
            [
                combined_reward(ref, hyp, self.config.reward_weights)
                for ref, hyp in zip(references, hypotheses, strict=True)
            ],
            device=self.device,
            dtype=torch.float32,
        )
        advantages = group_advantages(rewards, num_gen, self.config.advantage_eps)

        policy_log_probs = self._completion_log_probs(
            self.policy, features, prompt, completion_ids
        )
        with torch.no_grad():
            ref_log_probs = self._completion_log_probs(
                self.reference, features, prompt, completion_ids
            )
        old_log_probs = policy_log_probs.detach()
        mask = completion_mask_from_ids(completion_ids, self.eos_token_id)

        loss, mean_kl = grpo_loss(
            policy_log_probs,
            old_log_probs,
            ref_log_probs,
            advantages,
            mask,
            clip_eps=self.config.clip_eps,
            kl_beta=self.config.kl_beta,
        )

        batch_size = batch.input_features.size(0)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("train/reward", rewards.mean(), prog_bar=True, batch_size=batch_size)
        self.log("train/kl", mean_kl, batch_size=batch_size)
        self.log(
            "train/completion_len",
            mask.sum(dim=1).float().mean(),
            batch_size=batch_size,
        )
        return loss

    def train(self, mode: bool = True) -> "WhisperGRPOModule":
        """Keep the frozen reference model in eval mode at all times."""
        super().train(mode)
        self.reference.eval()
        return self

    def on_validation_epoch_start(self) -> None:
        """Reset the per-language WER accumulator."""
        self.val_metric.reset()

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Greedy-decode the batch and accumulate per-language WER."""
        prompt = decoder_prompt(
            self.policy, batch.input_features, self.config.task, batch.languages
        )
        completion_ids = self._generate(batch.input_features, prompt, sample=False)
        hypotheses = self.processor.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        for language, reference, hypothesis in zip(
            batch.languages, batch.references, hypotheses, strict=True
        ):
            self.val_metric.update(language, reference, hypothesis)

    def on_validation_epoch_end(self) -> None:
        """Log overall and per-language word and character error rates."""
        results = self.val_metric.compute()
        for metric, per_language in results.items():
            for language, rate in per_language.items():
                name = (
                    f"val/{metric}"
                    if language == "overall"
                    else f"val/{metric}_{language}"
                )
                prog_bar = metric == "wer" and language == "overall"
                self.log(name, rate, prog_bar=prog_bar, sync_dist=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the AdamW optimizer and a warmup + cosine schedule."""
        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
