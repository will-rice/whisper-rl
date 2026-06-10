"""Per-language word error rate aggregation.

Accumulates references and hypotheses bucketed by language so corpus-level WER
can be reported overall and broken down per language. Corpus WER (total edits
over total reference words) is used rather than a mean of per-utterance WERs,
which is the standard ASR reporting convention.
"""

from collections import defaultdict

import jiwer

from whisper_rl.rewards import normalize


class LanguageWER:
    """Bucket references/hypotheses by language and compute corpus WER."""

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
    def _corpus_wer(references: list[str], hypotheses: list[str]) -> float:
        """Corpus WER over normalized pairs, skipping empty references."""
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
        return float(jiwer.wer(refs, hyps))

    def compute(self) -> dict[str, float]:
        """Compute per-language and overall corpus WER.

        Returns:
            A mapping from each seen language code to its corpus WER, plus an
            ``"overall"`` key over all languages combined. Empty if nothing was
            accumulated.
        """
        results: dict[str, float] = {}
        all_refs: list[str] = []
        all_hyps: list[str] = []
        for language in sorted(self._references):
            refs = self._references[language]
            hyps = self._hypotheses[language]
            results[language] = self._corpus_wer(refs, hyps)
            all_refs.extend(refs)
            all_hyps.extend(hyps)
        if all_refs:
            results["overall"] = self._corpus_wer(all_refs, all_hyps)
        return results
