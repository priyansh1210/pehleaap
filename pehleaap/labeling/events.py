"""Turn-taking events from two channels' IPUs (plan §4.2 step 3). Thresholds: `LabelConfig`.

Rules, applied to each channel in turn (A) against the other (B):

* BC        - a short IPU of A while B holds the floor: at most `bc_max_s` and all backchannel
              vocabulary (or at most `bc_max_notext_s` without a transcript), with B speaking within
              `bc_context_s` before it and continuing within `bc_continue_s` after it.
* SHIFT     - A pauses and B's next non-BC IPU comes before A speaks again. B starting less than
              `terminal_overlap_s` before the pause still counts (early turn start, negative gap).
* HOLD      - A pauses and resumes before B says anything other than a backchannel.
              A pause that ends while B has been talking for longer is neither (A stopped under overlap).
* INTERRUPT - a non-BC IPU of A starting inside B's IPU (more than `terminal_overlap_s` before its end),
              at least `interrupt_min_s` long or `interrupt_min_words` words, where B either yields
              (stops within `yield_within_s` while A carries on) or the two compete for at least
              `competitive_overlap_s`.
* HOLD_REQ  - a hold phrase in A's IPU, then A stays quiet for at least `hold_req_min_silence_s`.

Backchannels are ignored throughout SHIFT/HOLD/INTERRUPT: the other's don't take the turn, and
one's own don't end one's pause (a "hmm" while the other replies doesn't make a SHIFT shorter). Rules
that need words (BC lexicon, HOLD_REQ, word counts, digit spans) wait for ASR; without it, only
durations are used. NOISE_ECHO (speakerphone leak) isn't detected yet, so speakerphone sessions
will show spurious BC/INTERRUPT events from echo.
"""

from pehleaap.labeling import lexicons
from pehleaap.labeling.config import LabelConfig
from pehleaap.labeling.ipu import IPU
from pehleaap.labeling.schema import BC, HOLD, HOLD_REQ, INTERRUPT, SHIFT, Event


def overlap(u: IPU, others: list[IPU]) -> float:
    return sum(max(0.0, min(u.end, o.end) - max(u.start, o.start)) for o in others)


def is_backchannel(u: IPU, other: list[IPU], cfg: LabelConfig) -> bool:
    if u.text:
        if u.dur > cfg.bc_max_s or lexicons.bc_type(u.text) is None:
            return False
    elif u.dur > cfg.bc_max_notext_s:
        return False
    spoke_before = any(o.start < u.start and o.end > u.start - cfg.bc_context_s for o in other)
    continues = any(o.end > u.end and o.start < u.end + cfg.bc_continue_s for o in other)
    return spoke_before and continues


def _digit_span(u: IPU, cfg: LabelConfig) -> bool | None:
    return None if u.text is None else lexicons.digit_run(u.text) >= cfg.digit_min_run


def _pause_events(ch: str, own_main: list[IPU], other_main: list[IPU], end_s: float, cfg: LabelConfig) -> list[Event]:
    out = []
    for i, u in enumerate(own_main):
        t0 = u.end
        t_own = own_main[i + 1].start if i + 1 < len(own_main) else None
        pause_s = (t_own if t_own is not None else end_s) - t0
        if pause_s < cfg.min_pause_s:
            continue
        common = dict(pause_s=pause_s, text=u.text, digit_span=_digit_span(u, cfg))

        if cfg.hold_req_min_silence_s <= pause_s and lexicons.is_hold_request(u.text):
            out.append(Event(ch, HOLD_REQ, t0, ipu_s=u.dur, **common))

        covering = [o for o in other_main if o.start <= t0 < o.end]
        if covering:
            started = min(o.start for o in covering)
            if t0 - started <= cfg.terminal_overlap_s:
                out.append(Event(ch, SHIFT, t0, gap_s=started - t0, **common))
            continue  # otherwise A stopped while B was mid-IPU: not a clean pause

        t_other = min((o.start for o in other_main if o.start > t0), default=None)
        if t_own is None and t_other is None:
            continue  # nobody speaks again
        if t_other is None or (t_own is not None and t_own < t_other):
            out.append(Event(ch, HOLD, t0, gap_s=t_own - t0, **common))
        else:
            out.append(Event(ch, SHIFT, t0, gap_s=t_other - t0, **common))
    return out


def _interrupts(ch: str, own_main: list[IPU], other_main: list[IPU], cfg: LabelConfig) -> list[Event]:
    out = []
    for u in own_main:
        host = next((o for o in other_main if o.start < u.start < o.end), None)
        if host is None:
            continue
        remaining = host.end - u.start  # how long B kept talking after A's onset
        if remaining <= cfg.terminal_overlap_s:
            continue  # early start of a normal turn transition
        if u.dur < cfg.interrupt_min_s and (u.n_words or 0) < cfg.interrupt_min_words:
            continue
        ov = overlap(u, other_main)
        yielded = remaining <= cfg.yield_within_s and u.end > host.end
        if yielded or ov >= cfg.competitive_overlap_s:
            out.append(
                Event(
                    ch,
                    INTERRUPT,
                    u.start,
                    ipu_s=u.dur,
                    overlap_s=ov,
                    yield_s=remaining if yielded else None,
                    text=u.text,
                    digit_span=_digit_span(u, cfg),
                )
            )
    return out


def extract_events(ipus: dict[str, list[IPU]], end_s: float, cfg: LabelConfig | None = None) -> list[Event]:
    """Events for both channels of one session, sorted by time.

    ipus: {channel: IPUs sorted by start}, exactly two channels (e.g. "agent", "customer").
    end_s: session length, which bounds the last pause of each channel.
    """
    cfg = cfg or LabelConfig()
    if len(ipus) != 2:
        raise ValueError(f"expected two channels, got {list(ipus)}")
    a, b = ipus
    other = {a: b, b: a}
    is_bc = {ch: [is_backchannel(u, ipus[other[ch]], cfg) for u in us] for ch, us in ipus.items()}
    main = {ch: [u for u, bc in zip(us, is_bc[ch]) if not bc] for ch, us in ipus.items()}

    events = []
    for ch, own in ipus.items():
        o = other[ch]
        for u, bc in zip(own, is_bc[ch]):
            if bc:
                ov = overlap(u, ipus[o])
                events.append(Event(ch, BC, u.start, ipu_s=u.dur, overlap_s=ov, bc_type=lexicons.bc_type(u.text), text=u.text))
        events += _pause_events(ch, main[ch], main[o], end_s, cfg)
        events += _interrupts(ch, main[ch], main[o], cfg)
    return sorted(events, key=lambda e: (e.t_event, e.channel, e.type))
