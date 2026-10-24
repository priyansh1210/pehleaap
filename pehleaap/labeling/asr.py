"""Speech recognition for the labeling engine (plan §4.2 step 2): words with timestamps, per channel.

Each speech segment (IPU) is transcribed on its own, padded by `pad_s` on both sides, instead of
the whole 25-minute track: Whisper invents text over long silences (a silent 2 s clip came back
as made-up Devanagari in a test), and per-segment results line up with the labels by
construction. Segments longer than `max_s` are cut into pieces. Words carry absolute times in
the track; `ipu.attach_words` gives each IPU the words whose midpoint falls inside it, so words
heard in the padding are dropped there.

The model runs privately on the GPU box or Kaggle, so no ASR provider receives the audio
(consent forms). Default: Whisper large-v3 (MIT) through transformers, so no extra dependency;
which model suits Hinglish best gets decided on the pilot recording (plan §4.2). Its output is
a draft: BC, HOLD_REQ and digit labels built on it are verified by a person (plan §4.2 step 4).
"""

import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pehleaap.labeling.ipu import IPU


@dataclass(frozen=True)
class AsrConfig:
    model: str = "openai/whisper-large-v3"
    language: str | None = "hi"  # Whisper's language token; Hinglish calls are mostly Hindi with English words. None: detect
    pad_s: float = 0.3
    max_s: float = 28.0  # Whisper reads 30 s windows
    batch_size: int = 8
    # Output cap per second of audio. Word timestamps keep every generated token's cross-attention, so a
    # repetition loop (common on short clips) can fill the GPU; fast Hindi is ~10 tokens/s.
    max_tokens_per_s: float = 20.0


def segments_for(ipus: list[IPU], end_s: float, pad_s: float, max_s: float) -> list[tuple[float, float]]:
    """The audio spans to transcribe: each IPU padded, clamped to the track, long ones cut into pieces <= max_s."""
    spans = []
    for u in ipus:
        n = max(1, int(np.ceil((u.end - u.start) / (max_s - 2 * pad_s))))
        edges = np.linspace(u.start, u.end, n + 1)
        spans += [(max(0.0, float(a) - pad_s), min(end_s, float(b) + pad_s)) for a, b in zip(edges[:-1], edges[1:])]
    return spans


def transcribe(audio: np.ndarray, sr: int, ipus: list[IPU], recognize, cfg: AsrConfig) -> list[dict]:
    """Words ({"start", "end", "word"}, seconds in the track) for one channel's IPUs.

    `recognize(clips, sr)` maps a list of mono float32 clips at `sr` to, per clip, a list of
    (start, end, text) relative to the clip; `WhisperASR` is one.
    """
    spans = segments_for(ipus, len(audio) / sr, cfg.pad_s, cfg.max_s)
    clips = [audio[int(round(a * sr)) : int(round(b * sr))].astype(np.float32) for a, b in spans]
    words = []
    for (a, b), found in zip(spans, recognize(clips, sr) if clips else []):
        for start, end, text in found:
            end = b - a if end is None else end  # Whisper sometimes gives no end for a word cut off at the clip's edge
            if text.strip():
                words.append({"start": round(a + start, 3), "end": round(a + max(end, start), 3), "word": text.strip()})
    return sorted(words, key=lambda w: w["start"])


class WhisperASR:
    """recognize() for `transcribe`, via transformers' Whisper pipeline with word timestamps."""

    def __init__(self, cfg: AsrConfig = AsrConfig(), device: str | None = None, precision: str = "auto"):
        import torch
        from transformers import pipeline

        from pehleaap.device import DTYPES, precision_for

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg, self.precision = cfg, precision_for(dev, precision)
        self.pipe = pipeline("automatic-speech-recognition", model=cfg.model, device=dev, dtype=DTYPES[self.precision])

        self.retried = 0  # segments decoded a second time because the first output looped

    def __call__(self, clips: list[np.ndarray], sr: int = 16000) -> list[list[tuple[float, float | None, str]]]:
        """Greedy decoding; segments whose text loops are decoded again with no 3-word sequence allowed twice.

        (transformers' own temperature fallback crashes together with word timestamps, checked with 5.20.2.)
        """
        results = self._decode(clips, sr, {})
        loops = [i for i, r in enumerate(results) if looping(" ".join(t for _, _, t in r))]
        if loops:
            again = self._decode([clips[i] for i in loops], sr, {"no_repeat_ngram_size": 3})
            for i, r in zip(loops, again):
                if compression_ratio(" ".join(t for _, _, t in r)) < compression_ratio(" ".join(t for _, _, t in results[i])):
                    results[i] = r
            self.retried += len(loops)
        return results

    def _decode(self, clips: list[np.ndarray], sr: int, extra: dict) -> list[list[tuple[float, float | None, str]]]:
        """Batches of similar length, each capped at max_tokens_per_s x its longest clip (Whisper's limit is 448)."""
        generate = {"task": "transcribe"} | ({"language": self.cfg.language} if self.cfg.language else {}) | extra
        order = np.argsort([len(c) for c in clips], kind="stable")
        results: list = [None] * len(clips)
        for i in range(0, len(order), self.cfg.batch_size):
            idx = order[i : i + self.cfg.batch_size]
            longest = max(len(clips[j]) for j in idx) / sr
            cap = min(440, int(self.cfg.max_tokens_per_s * longest) + 16)
            out = self.pipe([{"raw": clips[j], "sampling_rate": sr} for j in idx], return_timestamps="word", batch_size=len(idx),
                            generate_kwargs=generate | {"max_new_tokens": cap})
            for j, o in zip(idx, out):
                results[j] = [(w["timestamp"][0], w["timestamp"][1], w["text"]) for w in o.get("chunks", [])]
        return results


def compression_ratio(text: str) -> float:
    """How well the text's bytes compress: repetition loops compress far better than speech (Whisper's own check)."""
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw)) if raw else 0.0


def looping(text: str, threshold: float = 2.4) -> bool:
    """Whisper's threshold; short texts ("haan haan") can't reach it, since zlib adds a fixed overhead."""
    return compression_ratio(text) > threshold


def write_words(path: Path, words: dict[str, list[dict]], cfg: AsrConfig) -> None:
    """words.jsonl: one line per word with its channel, after a first line describing the ASR run."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"asr": asdict(cfg)}, ensure_ascii=False) + "\n")
        for channel, ws in words.items():
            for w in ws:
                f.write(json.dumps({"channel": channel, **w}, ensure_ascii=False) + "\n")


def read_words(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    """(ASR settings, words per channel) from a words.jsonl."""
    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    words: dict[str, list[dict]] = {}
    for w in lines[1:]:
        words.setdefault(w.pop("channel"), []).append(w)
    return lines[0]["asr"], words
