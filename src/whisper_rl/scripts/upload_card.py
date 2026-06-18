"""CLI to (re)generate a model card with W&B training curves for a run."""

import argparse
import logging

import wandb

from whisper_rl.cards import write_card

logging.basicConfig(level=logging.INFO)

ENTITY = "will-rice"
PROJECT = "whisper-rl"


def main() -> None:
    """Entry point for the ``upload-card`` console script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", type=str, help="Target HF model repo id.")
    parser.add_argument("run_id", type=str, help="8-character W&B run id.")
    parser.add_argument("--entity", default=ENTITY, type=str)
    parser.add_argument("--project", default=PROJECT, type=str)
    parser.add_argument("--license", default="mit", type=str)
    args = parser.parse_args()

    run = wandb.Api().run(f"{args.entity}/{args.project}/{args.run_id}")
    write_card(args.repo_id, run, args.license)


if __name__ == "__main__":
    main()
