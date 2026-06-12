"""Train Whisper with GRPO on word error rate."""

from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv
from git import Repo
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

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
    args = parser.parse_args()
    load_dotenv()

    config = Config()
    seed_everything(config.seed, workers=True)

    git_hash = Repo().head.object.hexsha[:7]
    model_name = config.base_model.split("/")[-1]
    experiment_name = f"{model_name}-grpo-{git_hash}"
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
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=str(experiment_path),
            monitor="val/wer",
            mode="min",
            save_top_k=1,
            filename="{step}-{val_wer:.3f}",
            auto_insert_metric_name=False,
        ),
    ]
    if logger:
        # LearningRateMonitor raises if the trainer has no logger.
        callbacks.append(LearningRateMonitor(logging_interval="step"))

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


if __name__ == "__main__":
    main()
