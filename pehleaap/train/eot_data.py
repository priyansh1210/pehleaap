"""Training data for the end-of-turn model: a clip store, a resumable sampler, and the batch builder.

Store: Smart Turn's training clips for the chosen languages, as one int16 file (`audio.int16`) that
DataLoader workers memory-map instead of copying, plus `index.csv` (id, language, label, offset,
length, split). Only the last 8 s of each clip is kept: that is all the model sees. 3% of clips,
picked by a hash of their id, form the dev split.

Batches: the sampler yields (epoch, row) keys in a seeded order, so a resumed run sees exactly the
batches it would have seen. The batch builder runs in the workers. Each clip goes through the
phone-line chain with probability `p_phone` (seeded by its key, so it is reproducible), using only
the training part of the noise files, then gets the same features as at evaluation time
(`WhisperMelOnnx.prepare`: last 8 s, left zero-padding, normalized Whisper log-mel).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from pehleaap.baselines.turn_models import SAMPLE_RATE, WINDOW, last_n_seconds

DEV_FRACTION = 0.03
INDEX_COLUMNS = ["id", "language", "label", "synthetic", "dataset", "offset", "length", "split", "shard"]
DEV_CLEAN, DEV_PHONE = -1, -2  # pseudo-epochs: dev clips scored clean, and through a fixed phone-line chain


def split_of(clip_id: str) -> str:
    return "dev" if int(hashlib.md5(clip_id.encode()).hexdigest()[:8], 16) < DEV_FRACTION * 16**8 else "train"


class EotStore:
    """Clips in `root`/audio.int16 with `root`/index.csv. Opens the audio lazily, so it pickles cheaply into workers."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index = pd.read_csv(self.root / "index.csv", dtype={"id": str}) if (self.root / "index.csv").exists() else pd.DataFrame(columns=INDEX_COLUMNS)
        self._audio = None

    def __getstate__(self):
        return {**self.__dict__, "_audio": None}

    @property
    def audio(self) -> np.ndarray:
        if self._audio is None:
            self._audio = np.memmap(self.root / "audio.int16", dtype=np.int16, mode="r")
        return self._audio

    def clip(self, row: int) -> np.ndarray:
        off, n = int(self.index.offset.iat[row]), int(self.index.length.iat[row])
        return self.audio[off : off + n].astype(np.float32) / 32768.0

    def rows(self, split: str, langs: tuple[str, ...]) -> np.ndarray:
        return np.flatnonzero((self.index.split == split).to_numpy() & self.index.language.isin(langs).to_numpy())

    def done_shards(self) -> set[str]:
        return set(self.index.shard.unique())

    def append(self, metas: list[dict], clips: list[np.ndarray], shard: str) -> None:
        """Add one shard's clips. Audio first, then the index rows, so a crash leaves only unindexed audio at the end."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._audio = None  # Windows can't truncate a file that is still memory-mapped
        audio_path = self.root / "audio.int16"
        end = int((self.index.offset + self.index.length).max()) if len(self.index) else 0
        with open(audio_path, "ab") as f:
            f.truncate(end * 2)  # drop audio a crashed run wrote without indexing it
            f.seek(end * 2)
            rows = []
            for m, x in zip(metas, clips):
                pcm = (np.clip(x[-WINDOW:], -1, 1) * 32767).astype(np.int16)
                f.write(pcm.tobytes())
                rows.append({"id": m["id"], "language": m["language"], "label": int(m["endpoint_bool"]), "synthetic": bool(m["synthetic"]),
                             "dataset": m["dataset"], "offset": end, "length": len(pcm), "split": split_of(m["id"]), "shard": shard})
                end += len(pcm)
        new = pd.DataFrame(rows, columns=INDEX_COLUMNS)
        new.to_csv(self.root / "index.csv", mode="a", header=not (self.root / "index.csv").exists(), index=False)
        self.index = pd.concat([self.index, new], ignore_index=True) if len(self.index) else new
        self._audio = None


class Keys(Dataset):
    """The sampler's keys pass straight through; the batch builder does the work."""

    def __getitem__(self, key):
        return key

    def __len__(self):
        raise TypeError("Keys has no length; use it with a sampler")


class EpochKeys(Sampler):
    """(epoch, row) keys for the whole run: each epoch is a seeded shuffle of `rows` (each repeated
    `repeat` times), cut to a whole number of batches. `start` skips the keys a resumed run already used."""

    def __init__(self, rows: np.ndarray, repeat: np.ndarray, epochs: int, batch_size: int, seed: int, start: int = 0):
        self.epoch_rows = np.repeat(rows, repeat)
        self.epoch_len = len(self.epoch_rows) // batch_size * batch_size
        self.epochs, self.seed, self.start = epochs, seed, start

    def __iter__(self):
        for epoch in range(self.start // self.epoch_len if self.epoch_len else self.epochs, self.epochs):
            order = np.random.default_rng([self.seed, epoch]).permutation(self.epoch_rows)[: self.epoch_len]
            skip = self.start - epoch * self.epoch_len
            for row in order[max(skip, 0) :]:
                yield epoch, int(row)

    def __len__(self):
        return max(self.epochs * self.epoch_len - self.start, 0)


class Augmenter:
    """collate_fn: (epoch, row) keys -> features (B, 80, 800), labels, languages, phone flags."""

    def __init__(self, store: EotStore, augment_config: str | Path, p_phone: float, seed: int):
        self.store, self.augment_config, self.p_phone, self.seed = store, str(augment_config), p_phone, seed
        self._tools = None

    def _setup(self):
        if self._tools is None:
            from transformers import WhisperFeatureExtractor

            from pehleaap.telephony.noise import NoiseBank
            from pehleaap.telephony.pipeline import load_config

            self._tools = (load_config(self.augment_config), NoiseBank(sr=SAMPLE_RATE, part="train"), WhisperFeatureExtractor(chunk_length=WINDOW // SAMPLE_RATE))
        return self._tools

    def __call__(self, keys: list[tuple[int, int]]) -> dict:
        from pehleaap.telephony.pipeline import phone_channel_many, sample_params

        cfg, bank, extractor = self._setup()
        clips = [self.store.clip(row) for _, row in keys]
        phone, params = [], []
        for i, (epoch, row) in enumerate(keys):
            rng = np.random.default_rng([self.seed, epoch + 2, row])  # pseudo-epochs are -1 and -2
            if epoch == DEV_PHONE or (epoch >= 0 and rng.random() < self.p_phone):
                phone.append(i)
                params.append(sample_params(cfg, rng, bank.names, speakerphone=False))
        if phone:
            for i, y in zip(phone, phone_channel_many([clips[i] for i in phone], params, SAMPLE_RATE, bank=bank)):
                clips[i] = y
        feats = extractor(
            [last_n_seconds(c) for c in clips], sampling_rate=SAMPLE_RATE, return_tensors="np",
            padding="max_length", max_length=WINDOW, truncation=True, do_normalize=True,
        ).input_features
        rows = [row for _, row in keys]
        is_phone = np.zeros(len(keys), dtype=bool)
        is_phone[phone] = True
        return {
            "features": torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)),
            "labels": torch.tensor(self.store.index.label.to_numpy()[rows], dtype=torch.float32),
            "language": self.store.index.language.to_numpy()[rows].tolist(),
            "phone": torch.from_numpy(is_phone),
        }
