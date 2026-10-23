import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_fake_pii_cards import card_for, luhn_valid  # noqa: E402


def _field(card: str, name: str) -> str:
    return re.search(rf"\| {re.escape(name)} \| (.+?) \|", card).group(1).replace(" ", "")


def test_luhn():
    assert luhn_valid("4539578763621486")
    assert not luhn_valid("4539578763621487")


def test_every_identifier_is_deliberately_invalid():
    rng = random.Random(0)
    for s in range(200):
        card = card_for(rng, s)
        assert _field(card, "Mobile")[0] in "12345"
        assert _field(card, "Alternate mobile")[0] in "12345"
        assert _field(card, "Aadhaar-format number")[0] in "01"
        assert _field(card, "PAN-format number")[3] == "X"
        assert not luhn_valid(_field(card, "Debit card"))
        assert _field(card, "PIN code").startswith("0")
