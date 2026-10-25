# Recruiting and outreach drafts

Ready-to-send drafts for week 1 (plan §7: book 15+ sessions; ask for HingCoS access). Fill the
`[brackets]` before sending. Keep the filled versions out of git: they hold names and contact details.

## 1. WhatsApp invite

**English**

> Hi [name]! I'm building speech software that helps phone voice-bots know when a person has
> *finished* speaking, instead of cutting them off mid-sentence. Bots today get this badly wrong in
> Hindi/Hinglish, so I'm recording real conversations to fix it.
>
> Would you take part? It's about **1 hour** on a voice call (Discord or Zoom) with another
> participant: two short role-plays from a card (one of you plays a customer-care agent, the other
> a customer, then you swap). Speak however you normally do: Hindi, English, mixed, all fine.
>
> - All numbers on the cards are made up; you never share real personal details.
> - No name is stored with the recording; you get an ID like SPK017.
> - Your audio stays private unless you separately agree to share it. You can withdraw any time.
> - [Thank-you: e.g. ₹___ / a treat / credit in the write-up]
>
> You need: earphones or a phone, a quiet-ish room, 18+. Free slots: [dates/times or booking link].
> Reply with a slot (and a friend who'd join you as your partner, if you like). I'll send the
> consent form to read first. Thank you! [your name]

**Hinglish**

> Hi [name]! Main ek speech software bana raha/rahi hoon jo phone par voice-bots ko samjhaata hai ki
> aapne baat *poori* kar li hai ya nahi, taaki bot beech mein na kaate. Hindi/Hinglish mein abhi ke
> bots yeh bahut galat karte hain, isliye main asli baatcheet record kar raha/rahi hoon.
>
> Kya aap help karenge? Lagbhag **1 ghanta**, ek voice call par (Discord ya Zoom) kisi aur ke saath:
> card se do chhote role-play (ek customer-care agent, ek customer, phir roles badal lete hain).
> Jaise aap normally bolte ho waise bolna hai: Hindi, English, mix, sab chalega.
>
> - Card ke saare numbers nakli hain; koi asli personal detail nahi bolni.
> - Recording ke saath naam store nahi hota; aapko ek ID milta hai jaise SPK017.
> - Aapki awaaz private rahegi jab tak aap alag se share karne ki permission na do. Kabhi bhi naam wapas le sakte ho.
> - [Thank-you: jaise ₹___ / treat / write-up mein credit]
>
> Chahiye: earphones ya phone, thoda shaant kamra, 18+ umar. Slots: [dates/times ya booking link].
> Ek slot bata do (aur chaho to ek dost jo aapke saath partner bane). Main pehle consent form bhej
> dunga/dungi padhne ke liye. Shukriya! [your name]

**Follow-up after someone says yes:** send `docs/consent_en.md` or `docs/consent_hi.md` (filled in), confirm the
slot and their partner, and add them to the booking sheet. Assign a speaker ID only once consent is signed.

## 2. Booking sheet (a private Google Sheet or Excel file, not in git)

`slot_date, slot_time, person, partner, phone_or_handle, consent_sent, consent_signed, speaker_id, device_plan, session_id, done`

- `device_plan`: aim for about 50% earphones, 30% handset, 20% speakerphone (recording_protocol.md §4).
- Week-1 exit criterion: **15 or more sessions booked**. That's about 30 people, or fewer if people do both roles with different partners.

## 3. HingCoS access request (email)

**Checked Oct 25: HingCoS is paid.** Access costs INR 50,000 + 18% GST, paid by demand draft to "Registrar, IIT Guwahati". It
also needs a signed form, emailed and posted with the draft to the contact on the request form (linked from
iitg.ac.in/eee/emstlab/HingCoS_Database/HingCoS.html). It's for research use only, with no
redistribution. It's read speech, which FLEURS already provides for free, so this email is only worth sending to ask
whether a student or academic copy can be had free. In that case, add a line asking for that to the draft below.

To: the corpus contact named on the HingCoS request form (EMST Lab, Dept. of EEE, IIT Guwahati)
Subject: Request for research access to the IITG-HingCoS corpus

> Dear [name],
>
> I'm working on turn-taking for Hindi-English (Hinglish) voice agents on telephone lines:
> detecting when a caller has finished speaking, as opposed to pausing mid-sentence, on 8 kHz
> phone audio. My project, pehleaap, builds an open benchmark and a training recipe for this
> (code: https://github.com/priyansh1210/pehleaap).
>
> I'd like to request access to the IITG-HingCoS corpus for non-commercial research. I'd use it to:
> - test robustness to real telephone recordings of code-mixed speech;
> - build "unfinished sentence" examples by cutting read sentences at word boundaries (for
>   example just before the final verb), to train and evaluate end-of-turn detection.
>
> I wouldn't redistribute the audio. Only aggregate results and derived model weights would be
> published, with the corpus cited as you prefer. I'm happy to sign a data-use agreement and to
> share the results with your lab.
>
> Could you tell me how to obtain access, and under which terms?
>
> Thank you for your time,
> [your name]
> [affiliation, if any] · [email]
