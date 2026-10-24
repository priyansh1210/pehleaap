import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pehleaap.audio.align import estimate_offset
from pehleaap.telephony.channel import agc, bandpass, conceal, gilbert_loss, late_packets
from pehleaap.telephony.echo import reverb, speakerphone_echo, synthetic_rir
from pehleaap.telephony.noise import NoiseBank, active_level_db, mix_at_snr, synthetic_noise
from pehleaap.telephony.pipeline import AugmentConfig, ChannelParams, load_config, phone_channel, phone_channel_many, sample_params

REPO = Path(__file__).resolve().parent.parent
SR = 16000


def _speechlike(seconds=3.0, seed=1, sr=SR):
    """Voiced bursts (pitch ~150 Hz, formant-ish smoothing) separated by silences."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 150 + 30 * np.sin(2 * np.pi * 1.7 * t)
    pulses = (np.diff(np.floor(np.cumsum(f0) / sr), prepend=0) > 0).astype(np.float32)
    voiced = np.convolve(pulses, np.hanning(16) * np.cos(np.arange(16) * 0.8), mode="same")
    gate = (np.sin(2 * np.pi * 1.3 * t) > 0).astype(np.float32)
    return (0.3 * voiced * gate + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


def _db(x):
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


def test_config_file_is_the_defaults():
    assert load_config(REPO / "configs" / "augment.yaml") == AugmentConfig()
    with pytest.raises(ValueError):
        AugmentConfig(codecs=("g711_mulaw", "speex"), codec_weights=(1, 1))


def test_full_chain_keeps_length_and_timing():
    """Labels are only valid if no stage moves the speech: within 1 ms, for every kind of draw."""
    bank = NoiseBank(root=None, sr=SR)
    cfg = AugmentConfig(p_reverb=0.5, p_agc=0.5, p_clip=0.3, p_jitter=0.5)
    rng = np.random.default_rng(0)
    clips = [_speechlike(2.0 + 0.5 * k, seed=k) for k in range(8)]
    params = [sample_params(cfg, rng, bank.names, speakerphone=False) for _ in clips]
    params[0] = replace(params[0], snr_bucket="clean", snr_db=None, noise=None)  # one plain draw at least
    for x, p, y in zip(clips, params, phone_channel_many(clips, params, SR, bank=bank)):
        assert len(y) == len(x) and y.dtype == np.float32
        assert abs(estimate_offset(bandpass(x, SR), y, SR, max_lag_s=0.1)) <= 0.001, p


def test_batched_equals_one_by_one_and_params_replay_from_json():
    bank = NoiseBank(root=None, sr=SR)
    clips = [_speechlike(1.5, seed=k) for k in range(3)]
    params = [replace(ChannelParams(seed=k), snr_bucket="10", snr_db=10.0, noise="synthetic:pink", loss_rate=0.05) for k in range(3)]
    batched = phone_channel_many(clips, params, SR, bank=bank)
    for x, p, y in zip(clips, params, batched):
        again = phone_channel(x, ChannelParams.from_dict(json.loads(json.dumps(p.to_dict()))), SR, bank=bank)
        np.testing.assert_allclose(y, again, atol=1e-6)  # G.711 has no state across clips


def test_output_rate_can_be_8k():
    x = _speechlike(1.0)
    assert len(phone_channel(x, ChannelParams(seed=0), SR, out_sr=8000)) == 8000


def test_active_level_ignores_silence_and_snr_is_exact():
    rng = np.random.default_rng(0)
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
    x = np.concatenate([tone, np.zeros(SR, dtype=np.float32)])
    assert active_level_db(x, SR) == pytest.approx(_db(tone), abs=0.05)  # plain RMS would read 3 dB lower
    assert active_level_db(np.zeros(SR, dtype=np.float32), SR) is None
    noise = synthetic_noise("pink", len(x), SR, rng)
    y = mix_at_snr(x, noise, 10.0, active_level_db(x, SR))
    assert _db(tone) - _db(y - x) == pytest.approx(10.0, abs=0.05)


@pytest.mark.parametrize("kind", ["white", "pink", "brown", "hum"])
def test_synthetic_noise_is_unit_rms(kind):
    y = synthetic_noise(kind, SR, SR, np.random.default_rng(0))
    assert len(y) == SR and np.sqrt(np.mean(y**2)) == pytest.approx(1.0, rel=1e-3)


def test_noise_bank_reads_files_and_loops_short_ones(tmp_path):
    sf.write(tmp_path / "a.wav", 0.05 * np.random.default_rng(0).standard_normal(SR // 2), SR)
    bank = NoiseBank(root=tmp_path, sr=SR)
    assert bank.names == ["a.wav"]
    seg = bank.segment("a.wav", 3 * SR, np.random.default_rng(1))
    assert len(seg) == 3 * SR and np.sqrt(np.mean(seg**2)) == pytest.approx(1.0, rel=1e-3)
    assert NoiseBank(root=tmp_path / "missing", sr=SR).names[0].startswith("synthetic:")


def test_reverb_adds_no_delay_and_drr_is_as_set():
    rir = synthetic_rir(SR, 0.5, 6.0, np.random.default_rng(0))
    assert rir[0] == 1.0 and 10 * np.log10(1.0 / np.sum(rir[1:] ** 2)) == pytest.approx(6.0, abs=0.01)
    x = _speechlike()
    assert abs(estimate_offset(x, reverb(x, rir), SR, max_lag_s=0.1)) <= 0.0005


def test_speakerphone_echo_delay_and_level():
    agent = _speechlike(4.0, seed=3)
    echo = speakerphone_echo(agent, SR, level_db=-20.0, delay_s=0.25, erle_db=15.0, drive=2.0, rt60_s=0.3, drr_db=8.0, rng=np.random.default_rng(0))
    assert len(echo) == len(agent)
    assert estimate_offset(agent, echo, SR, max_lag_s=1.0) == pytest.approx(0.25, abs=0.001)
    assert active_level_db(echo, SR) == pytest.approx(-35.0, abs=0.1)
    silent = speakerphone_echo(np.zeros(SR, dtype=np.float32), SR, -20.0, 0.25, 15.0, 2.0, 0.3, 8.0, np.random.default_rng(0))
    assert not silent.any()


def test_echo_reaches_the_customer_channel_through_the_chain():
    agent = np.concatenate([_speechlike(2.0, seed=5), np.zeros(2 * SR, dtype=np.float32)])
    customer = np.concatenate([np.zeros(2 * SR, dtype=np.float32), _speechlike(2.0, seed=6)])
    p = ChannelParams(seed=0, rt60_s=0.3, drr_db=6.0, echo_delay_s=0.2, erle_db=10.0, echo_drive=1.0)
    y = phone_channel(customer, p, SR, agent=agent)
    head = slice(0, int(1.9 * SR))  # agent talking, customer silent: only echo
    assert estimate_offset(agent[head], y[head], SR, max_lag_s=0.5) == pytest.approx(0.2, abs=0.002)
    assert active_level_db(y[head], SR) < active_level_db(y[2 * SR :], SR) - 6
    with pytest.raises(ValueError, match="agent track"):
        phone_channel(customer, p, SR)
    with pytest.raises(ValueError, match="NoiseBank"):
        phone_channel(customer, ChannelParams(seed=0, snr_bucket="5", snr_db=5.0, noise="synthetic:pink"), SR)


def test_bandpass_is_the_phone_band_and_zero_phase():
    t = np.arange(2 * SR) / SR
    gain = {f: _db(bandpass(np.sin(2 * np.pi * f * t), SR)[SR // 2 : -SR // 2]) - _db(np.sin(2 * np.pi * f * t)) for f in (100, 1000, 3900)}
    assert gain[1000] > -0.5 and gain[100] < -20 and gain[3900] < -10
    x = _speechlike()
    assert abs(estimate_offset(x, bandpass(x, SR), SR, max_lag_s=0.1)) <= 0.0005


def test_agc_levels_speech_and_lifts_noise_in_pauses():
    x = 10 ** (-40 / 20) * _speechlike(4.0) / 0.3  # quiet talker
    y = agc(x, SR, target_dbfs=-20.0, max_gain_db=30.0)
    assert active_level_db(y, SR) > active_level_db(x, SR) + 12
    assert active_level_db(agc(x, SR, -20.0, max_gain_db=6.0), SR) <= active_level_db(x, SR) + 6.1
    # After loud speech the gain is low; in the following pause it creeps back up, lifting the noise floor.
    floor = 1e-3 * np.random.default_rng(0).standard_normal(4 * SR).astype(np.float32)
    z = agc(np.concatenate([0.3 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR).astype(np.float32), floor]), SR, -20.0, 20.0)
    assert _db(z[-SR // 2 :]) > _db(z[SR + SR // 10 : SR + SR // 2]) + 6


def test_gilbert_loss_rate_and_bursts():
    lost = gilbert_loss(200_000, 0.05, 2.5, np.random.default_rng(0))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], lost.astype(int), [0]]))).reshape(-1, 2)
    assert lost.mean() == pytest.approx(0.05, abs=0.003)
    assert (edges[:, 1] - edges[:, 0]).mean() == pytest.approx(2.5, abs=0.15)
    assert not gilbert_loss(100, 0.0, 2.0, np.random.default_rng(0)).any()


def test_late_packets_queue_behind_a_spike():
    rates = [late_packets(50_000, 20.0, buf, np.random.default_rng(0)).mean() for buf in (60, 80, 120)]
    assert rates[0] > rates[1] > rates[2] and 0.01 < rates[1] < 0.06
    assert not late_packets(100, 0.0, 40.0, np.random.default_rng(0)).any()


def test_concealment_continues_the_pitch_then_fades():
    sr = 8000
    x = 0.3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr).astype(np.float32)
    lost = np.zeros(50, dtype=bool)
    lost[10] = True  # one 20 ms packet
    y = conceal(x, lost, sr)
    s = slice(10 * 160, 11 * 160)
    assert len(y) == len(x) and np.abs(y[s] - x[s])[:80].max() < 0.05  # the first 10 ms follow the waveform
    lost[20:30] = True  # 200 ms: concealment fades out by 60 ms, then silence
    y = conceal(x, lost, sr)
    assert np.abs(y[20 * 160 + 480 : 30 * 160]).max() == 0.0
    head = np.zeros(50, dtype=bool)
    head[0] = True  # nothing to repeat before the first packet
    assert not conceal(x, head, sr)[:160].any()


def test_sample_params_respects_switches():
    cfg = AugmentConfig()
    names = ["synthetic:pink"]
    draws = [sample_params(cfg, np.random.default_rng(i), names, speakerphone=False) for i in range(200)]
    assert not any(p.speakerphone for p in draws)
    assert all((p.snr_db is None) == (p.snr_bucket == "clean") for p in draws)
    assert all(len(p.codec) == 1 or (p.codec[1] == "g711_mulaw" and not p.codec[0].startswith("g711")) for p in draws)
    assert {p.label_fields()["snr_bucket"] for p in draws} == set(cfg.snr_buckets)
    on = sample_params(cfg, np.random.default_rng(0), names, speakerphone=True)
    assert on.speakerphone and on.erle_db is not None and on.drr_db is not None
