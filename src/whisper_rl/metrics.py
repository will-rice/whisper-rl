"""Per-language error rate aggregation.

Accumulates references and hypotheses bucketed by language so corpus-level
error rates can be reported overall and broken down per language. Corpus rates
(total edits over total reference units) are used rather than a mean of
per-utterance rates, which is the standard ASR reporting convention. Both word
error rate (WER) and character error rate (CER) are reported; CER is the
meaningful figure for languages without word spaces.
"""

from collections import defaultdict
from collections.abc import Callable

import jiwer

from whisper_rl.rewards import normalize

ERROR_RATES: dict[str, Callable[[list[str], list[str]], float]] = {
    "wer": jiwer.wer,
    "cer": jiwer.cer,
}


class LanguageErrorRate:
    """Bucket references/hypotheses by language and compute corpus WER and CER."""

    def __init__(self) -> None:
        self._references: dict[str, list[str]] = defaultdict(list)
        self._hypotheses: dict[str, list[str]] = defaultdict(list)

    def reset(self) -> None:
        """Clear all accumulated references and hypotheses."""
        self._references.clear()
        self._hypotheses.clear()

    def update(self, language: str, reference: str, hypothesis: str) -> None:
        """Record one ``(reference, hypothesis)`` pair for ``language``.

        Args:
            language: Whisper language code the clip belongs to.
            reference: Ground-truth transcription.
            hypothesis: Model-produced transcription.
        """
        self._references[language].append(reference)
        self._hypotheses[language].append(hypothesis)

    @staticmethod
    def _corpus_rate(
        references: list[str],
        hypotheses: list[str],
        rate: Callable[[list[str], list[str]], float],
    ) -> float:
        """Corpus error rate over normalized pairs, skipping empty references."""
        refs: list[str] = []
        hyps: list[str] = []
        for reference, hypothesis in zip(references, hypotheses, strict=True):
            normalized_ref = normalize(reference)
            if not normalized_ref:
                continue
            refs.append(normalized_ref)
            hyps.append(normalize(hypothesis))
        if not refs:
            return 0.0
        return float(rate(refs, hyps))

    def compute(self) -> dict[str, dict[str, float]]:
        """Compute per-language and overall corpus WER and CER.

        Returns:
            A mapping ``{"wer": {lang: rate, ..., "overall": rate}, "cer":
            {...}}``. Empty if nothing was accumulated.
        """
        if not self._references:
            return {}
        results: dict[str, dict[str, float]] = {name: {} for name in ERROR_RATES}
        all_refs: list[str] = []
        all_hyps: list[str] = []
        for language in sorted(self._references):
            refs = self._references[language]
            hyps = self._hypotheses[language]
            for name, rate in ERROR_RATES.items():
                results[name][language] = self._corpus_rate(refs, hyps, rate)
            all_refs.extend(refs)
            all_hyps.extend(hyps)
        for name, rate in ERROR_RATES.items():
            results[name]["overall"] = self._corpus_rate(all_refs, all_hyps, rate)
        return results
