"""Fetch MUSAN's noise part into data/external/noise/musan/ and check it against its manifest (plan §4.3).

MUSAN (openslr.org/17, CC BY 4.0: credit Snyder, Chen and Povey, 2015) is one 11 GB archive
(music, then speech, then noise). This streams it over four connections without saving it, keeps
only musan/noise/ (about 0.7 GB, with its LICENSE files) and stops when that section ends. About
75 min at the Dev box's 2.5 MB/s.

    python scripts/get_noise.py            # download, unless the copy on disk already verifies
    python scripts/get_noise.py --verify   # check only

The first run writes data/manifests/noise_musan.jsonl (commit it); later runs, on the GPU box
or Kaggle, check their copy against it.
"""

import argparse
import http.client
import io
import shutil
import sys
import tarfile
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.manifest import build_manifest, verify_manifest  # noqa: E402
from pehleaap.paths import MANIFEST_DIR  # noqa: E402
from pehleaap.telephony.noise import NOISE_DIR  # noqa: E402

# openslr's EU mirror. trmal.net is faster (3 MB/s vs 1.2 from the Dev box) but its TLS certificate had expired on 2026-10-24.
URL = "https://openslr.elda.org/resources/17/musan.tar.gz"
PREFIX = "musan/noise/"
DEST = NOISE_DIR / "musan"
MANIFEST = MANIFEST_DIR / "noise_musan.jsonl"


class ParallelDownload(io.RawIOBase):
    """Read-only stream of a URL, fetched as byte ranges over several connections at once.

    The mirror serves about 1 MB/s per connection; four connections gave 2.5 MB/s from the Dev box. Chunks
    are handed out in order; each one is retried on its own after a network error, so a dropped
    connection never restarts the download.
    """

    def __init__(self, url: str, workers: int = 4, chunk: int = 16 << 20, retries: int = 30):
        self.url, self.chunk, self.retries, self.workers = url, chunk, retries, workers
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") != "bytes":
                raise OSError("the server doesn't serve byte ranges")
        self.pool = ThreadPoolExecutor(workers)
        self.pending: deque = deque()
        self.next_start, self.pos, self.buf = 0, 0, memoryview(b"")
        self._fill()

    def _fill(self) -> None:
        while len(self.pending) < 2 * self.workers and self.next_start < self.size:
            end = min(self.next_start + self.chunk, self.size) - 1
            self.pending.append(self.pool.submit(self._fetch, self.next_start, end))
            self.next_start = end + 1

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if r.status == 206 and len(data) == end - start + 1:
                    return data
            except (OSError, http.client.HTTPException) as e:
                print(f"network error at {start / 1e9:.2f} GB ({e}); retrying", flush=True)
            time.sleep(min(2**attempt, 60))
        raise OSError(f"gave up on bytes {start}-{end} after {self.retries} tries")

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        if not len(self.buf):
            if not self.pending:
                return 0
            self.buf = memoryview(self.pending.popleft().result())
            self._fill()
        n = min(len(buf), len(self.buf))
        buf[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        self.pos += n
        return n

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().close()


def download() -> int:
    src = ParallelDownload(URL)
    tmp = DEST.with_name(DEST.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    n_files, seen, t0, last = 0, False, time.time(), 0.0
    with tarfile.open(fileobj=io.BufferedReader(src, 1 << 20), mode="r|gz") as tar:
        for m in tar:
            if not m.name.startswith(PREFIX):
                if seen:
                    break  # past the noise section
                if time.time() - last > 30:
                    print(f"skipping {m.name}: {src.pos / 1e9:.2f} of {src.size / 1e9:.1f} GB", flush=True)
                    last = time.time()
                continue
            seen = True
            rel = Path(m.name[len(PREFIX) :])
            if not m.isfile() or rel.is_absolute() or ".." in rel.parts:
                continue
            out = tmp / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as f, open(out, "wb") as g:
                shutil.copyfileobj(f, g)
            n_files += 1
    if not seen:
        raise SystemExit(f"no {PREFIX} in the archive")
    shutil.rmtree(DEST, ignore_errors=True)
    tmp.rename(DEST)
    print(f"{n_files} files, read {src.pos / 1e9:.2f} GB in {(time.time() - t0) / 60:.0f} min")
    return n_files


def check() -> bool:
    r = verify_manifest(MANIFEST)
    print(f"{MANIFEST.name}: {r['ok']} ok, {len(r['missing'])} missing, {len(r['changed'])} changed")
    return not r["missing"] and not r["changed"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="only check the copy on disk against the manifest")
    args = ap.parse_args()
    if args.verify:
        sys.exit(0 if MANIFEST.exists() and check() else 1)
    if MANIFEST.exists() and DEST.is_dir() and check():
        return
    download()
    if MANIFEST.exists():
        sys.exit(0 if check() else 1)
    print(f"wrote {build_manifest('external/noise/musan', MANIFEST.stem)}: commit it")


if __name__ == "__main__":
    main()
