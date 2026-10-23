"""Hinglish word lists for the label rules: backchannels, hold requests, number words.

ASR output can be romanized Hindi, Devanagari or English, so every list covers all three.
Text is normalized first: NFC, lowercase, punctuation to spaces, letter runs of 3+ collapsed
to 2 ("hmmmm" -> "hmm", "haaan" -> "haan").
"""

import re
import string
import unicodedata

_PUNCT = re.compile("[" + re.escape(string.punctuation + "।॥“”‘’…–—") + "]")
_RUNS = re.compile(r"(.)\1{2,}")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = _RUNS.sub(r"\1\1", _PUNCT.sub(" ", text))
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


# --- Backchannels: continuers ("go on") vs assessments ("got it / I agree") -----------------
CONTINUERS = {
    "haan", "haa", "ha", "han", "hm", "hmm", "mm", "mhm", "mhmm", "ji", "haanji", "hanji", "haji",
    "uh", "huh", "uhhuh", "yes", "yeah", "ya", "yup",
    "हाँ", "हां", "हा", "हम", "हम्म", "हं", "हूँ", "हूं", "जी", "हाँजी", "हांजी",
}  # fmt: skip
ASSESSMENTS = {
    "achha", "acha", "accha", "achcha", "achchha", "achhaa", "achaa", "theek", "thik", "ok", "okay", "okk",
    "sahi", "right", "bilkul", "sure", "great", "nice", "badhiya", "oh", "wah", "waah",
    "अच्छा", "ठीक", "ओके", "सही", "राइट", "बिल्कुल", "बढ़िया", "ओह", "वाह",
}  # fmt: skip
FUNCTION_WORDS = {"hai", "he", "है"}  # "theek hai", "sahi hai"; never a BC on their own

BC_VOCAB = CONTINUERS | ASSESSMENTS | FUNCTION_WORDS


def bc_type(text: str | None) -> str | None:
    """"continuer" / "assessment" if every word is backchannel vocabulary, else None."""
    toks = tokens(text or "")
    if not toks or any(t not in BC_VOCAB for t in toks):
        return None
    if any(t in ASSESSMENTS for t in toks):
        return "assessment"
    if any(t in CONTINUERS for t in toks):
        return "continuer"
    return None


# --- Hold requests ("ek minute", "ruko zara, card dhoondh raha hoon") ---------------------------
_HOLD_PHRASES = [
    r"(?:ek|1|one|do|2|two) (?:minute|min|minat|mint|second|sec|sekand|pal|moment)s?",
    r"just a (?:minute|min|moment|sec|second)",
    r"ruk(?:o|iye|iyega|ie|na|e)?(?: (?:jao|jaiye|jaaiye))?",
    r"hold (?:on|kar\S*|kij\S*)",
    r"wait",
    r"line (?:pe|par) rah\S*",
    r"dh(?:oo|u)n?dh?(?:\S*) (?:raha|rahi|rahe|ke|kar)",
    r"(?:check|dekh) (?:kar ?ke|ke) (?:bata|bol)\S*",
    r"abhi bata(?:ta|ti|te|unga|ungi|enge)",
    r"(?:एक|दो) (?:मिनट|मिनिट|सेकंड|सेकेंड|सेकण्ड|पल)",
    r"रुक(?:ो|िए|िये|िएगा|ना|े)?(?: (?:जाओ|जाइए|जाइये))?",
    r"होल्ड",
    r"वेट",
    r"ढू\S*ढ\S*",
    r"लाइन (?:पे|पर) रह\S*",
]
# Whole-token matches. \b doesn't work for Devanagari (vowel signs aren't \w), so use whitespace lookarounds.
HOLD_RE = re.compile("|".join(rf"(?<!\S)(?:{p})(?!\S)" for p in _HOLD_PHRASES))


def is_hold_request(text: str | None) -> bool:
    return bool(text and HOLD_RE.search(normalize(text)))


# --- Number words, for digit-dictation spans ------------------------------------------------
NUMBER_WORDS = {
    "shunya", "sunya", "zero", "ek", "do", "teen", "char", "chaar", "paanch", "panch", "chhe", "chhah", "chheh",
    "saat", "aath", "nau", "das", "gyarah", "barah", "baarah", "terah", "chaudah", "pandrah", "solah", "satrah",
    "atharah", "unnis", "bees", "tees", "chalis", "chaalis", "pachas", "pachaas", "saath", "sattar", "assi",
    "nabbe", "sau", "hazaar", "hazar", "lakh",
    "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "double", "triple",
    "शून्य", "ज़ीरो", "जीरो", "एक", "दो", "तीन", "चार", "पांच", "पाँच", "छह", "छः", "छे", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "बीस", "तीस", "चालीस",
    "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे", "सौ", "हज़ार", "हजार", "लाख", "डबल", "ट्रिपल",
}  # fmt: skip


def digit_run(text: str | None) -> int:
    """Longest run of consecutive number words, counting a numeral token as one per digit ("98 76" -> 4)."""
    best = run = 0
    for t in tokens(text or ""):
        if t.isdigit():
            run += len(t)
        elif t in NUMBER_WORDS:
            run += 1
        else:
            run = 0
        best = max(best, run)
    return best
