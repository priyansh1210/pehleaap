"""Write phone-line augmentation examples to listen to (plan §4.3).

Takes a few Hindi clips from Smart Turn's test set and writes, for each, the clean clip and the
customer channel under a few fixed conditions plus random draws from configs/augment.yaml. The
speakerphone example is a two-turn exchange (agent clip, then customer clip), so the agent's
echo is heard in the customer channel while the customer is quiet.

    python scripts/augment_examples.py                  # 3 clips -> runs/augment_examples/
    python scripts/augment_examples.py --clips 5 --random 6

Each wav has the same name in params.json, with the exact chain that made it.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.audio.io import save_audio  # noqa: E402
from pehleaap.data import smart_turn_testset  # noqa: E402
from pehleaap.paths import REPO_ROOT, RUNS_DIR  # noqa: E402
from pehleaap.telephony.noise import NoiseBank  # noqa: E402
from pehleaap.telephony.pipeline import ChannelParams, load_config, phone_channel, sample_params  # noqa: E402

SR = smart_turn_testset.SAMPLE_RATE


def fixed_conditions(noise: str) -> dict[str, ChannelParams]:
    base = ChannelParams(seed=0)
    return {
        "g711": base,
        "amr4.75_tandem": replace(base, codec=("amrnb_4.75", "g711_mulaw")),
        "noise5dB_amr7.95": replace(base, codec=("amrnb_7.95",), snr_bucket="5", snr_db=5.0, noise=noise, agc_target_dbfs=-20.0),
        "loss5pct_jitter": replace(base, loss_rate=0.05, loss_burst=2.0, jitter_ms=20.0, jitter_buffer_ms=70.0),
        "room_clipped": replace(base, rt60_s=0.5, drr_db=2.0, agc_target_dbfs=-16.0, clip_level=0.25),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--random", type=int, default=4, help="random chains per clip, drawn from the config")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "augment.yaml"))
    ap.add_argument("--out", default=str(RUNS_DIR / "augment_examples"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    bank = NoiseBank(sr=SR)
    noise = next((n for n in bank.names if "street" in n or "traffic" in n), bank.names[min(1, len(bank.names) - 1)])
    metas, clips = smart_turn_testset.read_shard(smart_turn_testset.shard_paths()[0], ["hin"])
    rng = np.random.default_rng(0)
    out = Path(args.out)
    params = {}
    for k in range(args.clips):
        x, clip_id = clips[k], metas[k]["id"]
        save_audio(out / f"{k}_clean.wav", x, SR)
        conds = fixed_conditions(noise)
        conds |= {f"random{j}": sample_params(cfg, rng, bank.names, speakerphone=False) for j in range(args.random)}
        for name, p in conds.items():
            save_audio(out / f"{k}_{name}.wav", phone_channel(x, p, SR, bank=bank), SR)
            params[f"{k}_{name}.wav"] = {"clip": clip_id, **p.to_dict()}

        # Speakerphone: the agent says the next clip, then the customer answers with this one.
        agent_clip = clips[(k + 1) % len(clips)]
        pad = np.zeros(int(0.8 * SR), dtype=np.float32)
        agent = np.concatenate([agent_clip, pad, np.zeros_like(x)])
        customer = np.concatenate([np.zeros_like(agent_clip), pad, x])
        p = replace(ChannelParams(seed=k), rt60_s=0.4, drr_db=4.0, echo_delay_s=0.3, erle_db=12.0, echo_drive=2.0)
        save_audio(out / f"{k}_speakerphone_stereo.wav", np.stack([agent, phone_channel(customer, p, SR, agent=agent)]), SR)
        params[f"{k}_speakerphone_stereo.wav"] = {"clip": clip_id, "agent_clip": metas[(k + 1) % len(clips)]["id"], **p.to_dict()}

    (out / "params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    print(f"{len(params) + args.clips} files in {out} (noise: {noise}; speakerphone files: left = agent, right = customer)")


if __name__ == "__main__":
    main()
