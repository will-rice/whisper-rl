"""Score Whisper checkpoints on the shared evaluation slice.

A GRPO run reports the error rates of the model it produced, which says nothing
on its own -- a word error rate of 0.87 is either a strong result on a
low-resource language or a broken model, and only the baseline distinguishes
them. This scores any number of checkpoints over the identical clips so the
difference against ``openai/whisper-tiny`` is measurable rather than asserted.

The evaluation slice is materialized with ``take`` and never shuffled, so every
model here sees the same clips in the same order, and decoding is greedy. The
comparison is therefore exact rather than approximate.
"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv
from lightning import Trainer, seed_everything

from whisper_rl.config import Config
from whisper_rl.datasets import SpeechDataModule
from whisper_rl.lightning_module import WhisperGRPOModule
from whisper_rl.modeling import build_processor

# Clips pulled from the evaluation split. The slice spans every locale in the
# dataset, so this divides across all of them -- too small a number leaves
# single-digit clips per language and per-language rates stop meaning anything.
EVAL_SAMPLES = 4096


def main() -> None:
    """Entry point for the ``evaluate`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = ArgumentParser(description="Score Whisper checkpoints on shared clips.")
    parser.add_argument(
        "models",
        nargs="+",
        help="Hub ids or local paths. The first is treated as the baseline.",
    )
    parser.add_argument("--output", default=Path("eval.json"), type=Path)
    parser.add_argument("--num_devices", default=1, type=int)
    args = parser.parse_args()
    load_dotenv()

    results = {model: evaluate(model, args.num_devices) for model in args.models}
    report = {
        "baseline": args.models[0],
        "eval_samples": EVAL_SAMPLES,
        "results": results,
        "deltas": {
            model: deltas(results[args.models[0]], results[model])
            for model in args.models[1:]
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    logging.info("Wrote %s", args.output)
    log_summary(report)


def evaluate(model: str, num_devices: int) -> dict[str, dict[str, float]]:
    """Score one checkpoint over the evaluation slice.

    Args:
        model: Hub id or local path of a Whisper checkpoint.
        num_devices: Devices to run validation across.

    Returns:
        Per-language and overall corpus WER and CER, as
        ``{"wer": {lang: rate, ..., "overall": rate}, "cer": {...}}``.
    """
    logging.info("Evaluating %s", model)
    config = Config(base_model=model, max_eval_samples=EVAL_SAMPLES)
    seed_everything(config.seed, workers=True)
    processor = build_processor(config)
    module = WhisperGRPOModule(config, processor)
    datamodule = SpeechDataModule(config, processor)
    trainer = Trainer(devices=num_devices, logger=False, enable_checkpointing=False)
    trainer.validate(module, datamodule=datamodule)
    return module.val_metric.compute()


def deltas(
    baseline: dict[str, dict[str, float]], candidate: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Subtract baseline rates from candidate rates.

    Negative values are improvements, since lower error rates are better.

    Args:
        baseline: Rates from the baseline checkpoint.
        candidate: Rates from the checkpoint being compared.

    Returns:
        The same nested shape, holding ``candidate - baseline`` for every
        language both scored.
    """
    return {
        metric: {
            language: candidate[metric][language] - rate
            for language, rate in per_language.items()
            if language in candidate[metric]
        }
        for metric, per_language in baseline.items()
    }


def log_summary(report: dict) -> None:
    """Log overall rates and the count of languages each candidate improved."""
    for model, rates in report["results"].items():
        logging.info(
            "%s: WER %.4f  CER %.4f",
            model,
            rates["wer"]["overall"],
            rates["cer"]["overall"],
        )
    for model, delta in report["deltas"].items():
        improved = sum(
            1 for lang, d in delta["wer"].items() if lang != "overall" and d < 0
        )
        total = sum(1 for lang in delta["wer"] if lang != "overall")
        logging.info(
            "%s: WER %+.4f overall, improved %d of %d languages",
            model,
            delta["wer"]["overall"],
            improved,
            total,
        )


if __name__ == "__main__":
    main()
