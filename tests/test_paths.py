import pytest

from pehleaap import paths

ENCODERS = {
    "/usr/bin/ffmpeg": " A....D libgsm  GSM\n A....D libopus  Opus\n A....D pcm_mulaw  PCM mu-law\n",
    "/tmp/static/ffmpeg": " A....D libopencore_amrnb  OpenCORE AMR-NB\n A....D pcm_mulaw  PCM mu-law\n",
}


@pytest.fixture
def kaggle_like(monkeypatch):
    """Two builds, as on Kaggle: the system ffmpeg first, the static one from PEHLEAAP_FFMPEG_EXTRA."""
    monkeypatch.setattr(paths, "_ffmpeg_candidates", lambda: list(ENCODERS))
    monkeypatch.setattr(paths, "_ffmpeg_encoders", ENCODERS.get)


def test_each_encoder_comes_from_the_first_build_that_has_it(kaggle_like):
    assert paths.ffmpeg_bin() == "/usr/bin/ffmpeg"
    assert paths.ffmpeg_bin("libgsm") == "/usr/bin/ffmpeg"
    assert paths.ffmpeg_bin("pcm_mulaw") == "/usr/bin/ffmpeg"
    assert paths.ffmpeg_bin("libopencore_amrnb") == "/tmp/static/ffmpeg"


def test_missing_encoder_names_the_fix(kaggle_like):
    with pytest.raises(FileNotFoundError, match="PEHLEAAP_FFMPEG_EXTRA"):
        paths.ffmpeg_bin("libmp3lame")


def test_extra_builds_are_searched_after_the_defaults(monkeypatch):
    monkeypatch.setenv("PEHLEAAP_FFMPEG", "/a/ffmpeg")
    monkeypatch.setenv("PEHLEAAP_FFMPEG_EXTRA", "/b/ffmpeg" + paths.os.pathsep + "/a/ffmpeg")
    found = paths._ffmpeg_candidates()
    assert found[0] == "/a/ffmpeg" and found[-1] == "/b/ffmpeg" and found.count("/a/ffmpeg") == 1
