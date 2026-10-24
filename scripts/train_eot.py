"""Fine-tune Smart Turn v3.2 for the phone line; see pehleaap/train/train_eot.py for what it does and writes.

    python scripts/prepare_eot_train.py                                    # once: the training clips
    python scripts/train_eot.py --out eot_v0                               # train (rerun the same command to resume)
    python scripts/train_eot.py --out smoke --max-steps 5 --set micro_batch=8 batch_size=16 num_workers=2 dev_limit=16
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pehleaap.train.train_eot import main  # noqa: E402

if __name__ == "__main__":  # DataLoader workers re-import this file on Windows
    main()
