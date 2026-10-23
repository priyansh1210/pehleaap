"""Generate printable fake-PII cards, one per session, so nobody reads out real numbers.

Every identifier is generated in its real-world *format* but made deliberately invalid, so it
can't belong to a real person:
  mobile         10 digits starting 1-5 (Indian mobile numbers start 6-9)
  Aadhaar-format 12 digits starting 0 or 1 (real Aadhaar numbers never start with 0/1)
  PAN-format     4th character X (not a valid PAN holder-type code)
  card           16 digits that fail the Luhn check
  PIN code       6 digits starting 0 (Indian PIN codes start 1-9)

    python scripts/make_fake_pii_cards.py --sessions 60
"""

import argparse
import random
import string
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "fake_pii_cards"

FIRST = ["Rahul", "Priya", "Amit", "Neha", "Vikas", "Pooja", "Arjun", "Sneha", "Rohit", "Anjali", "Karan", "Divya"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Yadav", "Patel", "Jain", "Mishra", "Agarwal", "Chauhan"]
CITIES = ["Jaipur", "Lucknow", "Indore", "Pune", "Noida", "Bhopal", "Patna", "Nagpur", "Kanpur", "Ahmedabad"]
SOCIETIES = ["Shanti Vihar", "Green Park", "Gokul Dham", "Sunrise Apartments", "Ashok Nagar", "Om Residency"]
LANDMARKS = ["Hanuman mandir ke paas", "SBI ATM ke saamne", "petrol pump ke peeche", "D-Mart ke paas", "school ke bagal mein"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def luhn_valid(number: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def fake_card(rng: random.Random) -> str:
    while True:
        n = rng.choice("3456") + digits(rng, 15)
        if not luhn_valid(n):
            return " ".join(n[i : i + 4] for i in range(0, 16, 4))


def card_for(rng: random.Random, session: int) -> str:
    mobile = rng.choice("12345") + digits(rng, 9)
    alt_mobile = rng.choice("12345") + digits(rng, 9)
    aadhaar = rng.choice("01") + digits(rng, 11)
    pan = "".join(rng.choice(string.ascii_uppercase) for _ in range(3)) + "X" + rng.choice(string.ascii_uppercase) + digits(rng, 4) + rng.choice(string.ascii_uppercase)
    amount = rng.choice([1850, 2499, 3275, 4560, 7999, 12450, 15500, 18750, 23400])
    late_fee = rng.choice([350, 500, 590, 750])
    dob = f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(1965, 2003)}"
    return f"""# Fake details: session {session:03d}

**These are made up. Never use your real numbers.** Everything here is deliberately invalid.

| Item | Value |
|---|---|
| Name | {rng.choice(FIRST)} {rng.choice(LAST)} |
| Date of birth | {dob} |
| Mobile | {mobile[:5]} {mobile[5:]} |
| Alternate mobile | {alt_mobile[:5]} {alt_mobile[5:]} |
| Aadhaar-format number | {aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]} |
| PAN-format number | {pan} |
| Debit card | {fake_card(rng)} |
| Loan account | LN{digits(rng, 10)} |
| Policy number | POL-{digits(rng, 4)}-{digits(rng, 6)} |
| Order ID | OD{digits(rng, 12)} |
| Recharge transaction ID | TXN{rng.choice(string.ascii_uppercase)}{digits(rng, 3)}{rng.choice(string.ascii_uppercase)}{digits(rng, 5)} |
| Address | Flat {rng.randint(101, 1204)}, Tower {rng.choice("ABCDE")}, {rng.choice(SOCIETIES)}, {rng.choice(LANDMARKS)}, {rng.choice(CITIES)} |
| PIN code | 0{digits(rng, 5)} |
| EMI / bill amount | ₹{amount:,} |
| Late fee | ₹{late_fee} |
| Last transaction | ₹{rng.randint(120, 9800):,} at {rng.choice(["Swiggy", "Amazon", "petrol pump", "DMart", "Zomato", "Flipkart"])} |
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in range(1, args.sessions + 1):
        (OUT_DIR / f"session_{s:03d}.md").write_text(card_for(rng, s), encoding="utf-8")
    print(f"wrote {args.sessions} cards to {OUT_DIR}")


if __name__ == "__main__":
    main()
