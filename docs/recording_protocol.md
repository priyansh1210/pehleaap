# Recording protocol

The goal is **two-channel** Hinglish conversations, with each speaker on their own track and
both tracks on one clock, covering the behaviour voice agents get wrong: verb-final pauses,
"haan/hmm" backchannels, hold requests, digit dictation, interruptions and speakerphone echo.

Targets: 40+ adult speakers, about 25–30 pairs, 2 sessions per pair of 20–25 min, roles
swapped between sessions. That gives about 17–20 h, plus 1–2 h over real phone lines (test set only).

---

## 1. Pilot (week 1, one pair, about 45 min): choose the recording tool

Try both tools with the **same pair in the same room**, each person on their own laptop or phone
**with headphones on**. Being in the same room lets a single clap reach both microphones,
which is how alignment is measured.

### Option A: Craig on Discord (multitrack FLAC)
1. Create a private Discord server, add the Craig bot (craig.chat), and both people join one voice channel.
2. In each person's Discord settings, under **Voice & Video**: Noise suppression **None**,
   Echo cancellation **off**, Automatic gain control **off**, Input sensitivity **manual** at a low threshold.
3. Type `/join` in the channel to start Craig. It announces that it is recording.
4. **Clap** once, sharply, with both people silent. Talk for 3 min. **Clap** again. Talk for 10 more min. **Clap**.
5. Run `/stop`, then download **Multi-track → FLAC**.

### Option B: Zoom local recording, one file per participant
1. On the host's desktop client: Settings → Recording → tick **"Record a separate audio file for each participant"**.
2. Settings → Audio: enable **"Original sound for musicians"** and turn **off** "Echo cancellation"
   inside it. Background noise suppression **Low/off**. Both participants do this.
3. Same clap sequence as above. Record locally, then find the `Audio Record` folder with one file per person.

### Pass/fail check (for each tool)
```
python scripts/check_alignment.py <speakerA_track> <speakerB_track>
```
- **Drift ≤ 10 ms** between the first and last clap → the tool keeps both tracks on one clock. **Required.**
- **Start offset**: in a same-room test, this is the difference in capture plus network delay
  between the two clients, typically tens of ms. Write it down. It doesn't have to be 0, because
  both tools timestamp audio as it reaches *them*, which is also how a voice-agent server sees the call.
- Then listen to each track: there should be no noise-suppression "pumping", and no clipping or AGC swells.

Pick the tool that passes with the least audio processing. If **neither** passes, fall back to the
`recorder/` aiortc relay (see the plan, §4.1).

---

## 2. Before every session

- [ ] Both participants have **signed consent** (`consent_en.md` / `consent_hi.md`) and have speaker IDs.
- [ ] Each has a printed **scenario card** (`scenario_cards.md`) and the session's **fake-PII card**
      (`fake_pii_cards/session_NNN.md`). Remind them: **never say real numbers**.
- [ ] Device condition decided (see §4) and noted in `sessions.csv`.
- [ ] Quiet-ish room is fine. Real background (fan, TV, traffic) is welcome; music is not.
- [ ] The recording tool settings match §1 (noise suppression off, echo cancellation off).

## 3. Session flow (about 25 min)

| Time | Step |
|---|---|
| 0:00 | Start recording. **Sync beep**: the "agent" plays a short beep from a phone held near their mic, while the "customer" briefly takes off their headphones / un-mutes their speaker so the beep plays out loud and is picked up by the customer's mic too. Then headphones back on. |
| 0:30 | Warm-up: 1 min of chat about anything. |
| 1:30 | **Scenario 1** (about 8 min), following the cards. |
| 9:30 | **Scenario 2** (about 8 min). |
| 17:30 | **Free chat** (about 5 min): plans, cricket, food, films. Natural conversation only, no script. |
| 22:30 | **Sync beep** again, the same way. Stop recording. |

The two beeps let `check_alignment.py` confirm there was no clock drift in every session, not
just the pilot. In remote sessions the absolute offset includes network delay, but the
**drift** must still be ≤ 10 ms.

**Rules for the conversation** (also printed on the cards):
- Speak naturally, mixing Hindi and English the way you normally would. Don't read the card word for word.
- The **customer** must: ask the agent to wait at least **twice** ("ek minute…", "ruko zara, card dhoondh raha/rahi hoon")
  and then actually go quiet for a few seconds; **interrupt** the agent at least **twice** while the agent is
  in the middle of a long explanation; read numbers from the fake card **in groups, with pauses**.
- The **agent** must: give at least one **long explanation** (terms, process, charges) to give the customer
  something to interrupt; use natural "haan ji", "achha", "hmm" while the customer talks.
- Both: if you mishear, just ask again. Laughter, fillers and false starts are all good data.

## 4. Device conditions (logged per session)

| Condition | Share | How |
|---|---|---|
| `earphones` | ~50% | Wired or Bluetooth earphones with mic |
| `handset` | ~30% | Phone held to the ear (Discord/Zoom mobile app) |
| `speakerphone` | ~20% | Phone or laptop on the table with **loudspeaker on** and **echo cancellation off**, so the other person's voice leaks into this mic. This is the real echo data. |

## 5. Files and metadata

Save each session as:
```
data/raw/sessions/S017/
  S017_SPK031_agent.flac
  S017_SPK032_customer.flac
  S017_meta.json        # tool, devices, scenarios, beep times if known
```
Add one row per session to `data/raw/sessions.csv`:
`session_id, date, tool, spk_agent, spk_customer, device_agent, device_customer, scenario_1, scenario_2, pii_card, environment, notes`

…and per speaker to `data/raw/speakers.csv` (private, never published with names):
`speaker_id, age_range, gender, home_state, languages, codemix_self_rating(1-5), consent_public_audio(yes/no), consent_date`

After each recording day:
```
python scripts/check_alignment.py data/raw/sessions/S017/S017_SPK031_agent.flac data/raw/sessions/S017/S017_SPK032_customer.flac
python scripts/label_session.py data/raw/sessions/S017     # automatic turn-taking labels -> data/labels/S017/
python -c "from pehleaap.manifest import build_manifest; build_manifest('raw')"
```
Then commit `data/manifests/raw.jsonl` (checksums only, no audio) and sync `data/raw` to the GPU box.

## 6. Real-phone (PSTN) sessions: test set only

The same scenario cards and the same flow, but the call is a real phone call bridged by a telephony provider
with **dual-channel recording** (each leg on its own channel). There are about 6 pairs, and these sessions never go into
training. The provider is set up in week 1 (Twilio `record-from-answer-dual` first; Exotel if its KYC allows).
