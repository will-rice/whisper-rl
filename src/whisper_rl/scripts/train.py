"""Train Whisper with GRPO on word error rate."""

import logging
from argparse import ArgumentParser
from pathlib import Path

import wandb
from dotenv import load_dotenv
from git import Repo
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from transformers import WhisperProcessor

from whisper_rl.cards import write_card
from whisper_rl.config import Config
from whisper_rl.datasets import SpeechDataModule
from whisper_rl.lightning_module import WhisperGRPOModule
from whisper_rl.modeling import build_processor


def main() -> None:
    """Entry point for the ``train`` console script."""
    parser = ArgumentParser(description="Finetune Whisper with GRPO on WER.")
    parser.add_argument("--project", default="whisper-rl", type=str)
    parser.add_argument("--num_devices", default=1, type=int)
    parser.add_argument("--log_root", default="logs", type=Path)
    parser.add_argument("--checkpoint_path", default=None, type=Path)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument(
        "--run_suffix",
        default="",
        type=str,
        help="Suffix for the experiment name, e.g. a sweep configuration.",
    )
    args = parser.parse_args()
    load_dotenv()

    config = Config()
    seed_everything(config.seed, workers=True)

    git_hash = Repo().head.object.hexsha[:7]
    model_name = config.base_model.split("/")[-1]
    experiment_name = f"{model_name}-grpo-{git_hash}"
    if args.run_suffix:
        experiment_name = f"{experiment_name}-{args.run_suffix}"
    experiment_path = args.log_root / experiment_name
    experiment_path.mkdir(exist_ok=True, parents=True)

    processor = build_processor(config)
    datamodule = SpeechDataModule(config, processor)
    module = WhisperGRPOModule(config, processor)

    logger: WandbLogger | bool = (
        False
        if args.no_wandb
        else WandbLogger(
            project=args.project, name=experiment_name, save_dir=str(args.log_root)
        )
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(experiment_path),
        monitor="val/reward",
        mode="max",
        save_top_k=1,
        filename="{step}-{val/wer:.3f}",
        auto_insert_metric_name=False,
    )
    callbacks: list[Callback] = [checkpoint]
    if logger:
        # LearningRateMonitor raises if the trainer has no logger.
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if not args.fast_dev_run:
        callbacks.append(PushBestToHub(experiment_name, processor))

    trainer = Trainer(
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        devices=args.num_devices,
        gradient_clip_val=config.grad_clip,
        val_check_interval=config.val_check_interval,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=str(experiment_path),
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.checkpoint_path)


class PushBestToHub(Callback):
    """Push the policy to the Hugging Face Hub whenever validation reward improves.

    The live module already holds the weights that ModelCheckpoint is about to
    save as the new best, so they are uploaded directly — in standard
    ``transformers`` format, without the frozen reference model and optimizer
    state bundled in the Lightning checkpoint. A crashed run keeps its
    best-so-far model on the Hub.
    """

    def __init__(self, repo_name: str, processor: WhisperProcessor) -> None:
        self.repo_name = repo_name
        self.processor = processor
        self.best_reward = float("-inf")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Upload the policy and processor after a new best validation reward."""
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        reward = trainer.callback_metrics.get("val/reward")
        if reward is None or float(reward) <= self.best_reward:
            return
        self.best_reward = float(reward)
        assert isinstance(pl_module, WhisperGRPOModule)
        pl_module.policy.push_to_hub(self.repo_name)  # ty: ignore[invalid-argument-type]
        self.processor.push_to_hub(self.repo_name)
        logging.info(
            "Pushed new best (val/reward=%.4f) to the Hub as %s",
            self.best_reward,
            self.repo_name,
        )
        self._update_card(trainer)

    def _update_card(self, trainer: Trainer) -> None:
        """Regenerate the model card from the live W&B run, if one exists.

        Wrapped so a transient W&B or Hub failure logs a warning rather than
        crashing a long training run.
        """
        if not isinstance(trainer.logger, WandbLogger):
            return
        run_path = "/".join(trainer.logger.experiment.path)
        try:
            write_card(self.repo_name, wandb.Api().run(run_path))
        except Exception as error:  # pragma: no cover - network/W&B issues
            logging.warning("Skipped model-card update: %s", error)


if __name__ == "__main__":
    main()
