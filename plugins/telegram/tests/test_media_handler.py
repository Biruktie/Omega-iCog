import io
import os
import sys
import types

TELEGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TELEGRAM_DIR))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)
sys.path.insert(1, TELEGRAM_DIR)

from unittest.mock import AsyncMock, patch

import media_handler as mh
import speech_language as sl


def _fake_gtts(audio=b"google-mp3", expected_text=None, expected_lang=None):
    """Create a fake gTTS module that writes predictable audio."""
    fake_gtts = types.ModuleType("gtts")

    class FakeGTTS:
        def __init__(self, text, lang, timeout=None):
            if expected_text is not None:
                assert text == expected_text
            if expected_lang is not None:
                assert lang == expected_lang
            self.timeout = timeout

        def write_to_fp(self, buf):
            buf.write(audio)

    fake_gtts.gTTS = FakeGTTS
    return fake_gtts


def _fake_edge(communicate_class):
    """Create a fake edge_tts module."""
    fake_edge = types.ModuleType("edge_tts")
    fake_edge.Communicate = communicate_class
    return fake_edge


def test_no_image_returns_marker():
    mh.clear_pending()
    mh.set_pending_media(None)
    assert mh.describe_image("anything") == "[NO_IMAGE: nothing is attached to describe]"


def test_describe_memoizes_per_turn():
    calls = {"n": 0}
    orig = mh._call_vision_model

    def fake(image_parts, prompt):
        calls["n"] += 1
        return "a red panda"

    mh._call_vision_model = fake
    try:
        mh.set_pending_media([{"type": "image_url",
                               "image_url": {"url": "data:image/jpeg;base64,AAAA"}}])
        r1 = mh.describe_image("")
        r2 = mh.describe_image("")
        assert r1 == r2 == "[IMAGE DESCRIPTION]\na red panda", r1
        assert calls["n"] == 1, calls["n"]

        # A reply's clear_pending() must NOT blind describe-image mid-turn.
        mh.clear_pending()
        assert mh.describe_image("") == "[IMAGE DESCRIPTION]\na red panda"
        # A new non-image message makes the prior image stale.
        mh.set_pending_media(None)
        assert mh.describe_image("") == "[NO_IMAGE: nothing is attached to describe]"
    finally:
        mh._call_vision_model = orig
        mh.clear_pending()
        mh.set_pending_media(None)


def test_describe_never_raises():
    def boom(image_parts, prompt):
        raise RuntimeError("network down")

    orig = mh._call_vision_model
    mh._call_vision_model = boom
    try:
        mh.set_pending_media([{"type": "image_url",
                               "image_url": {"url": "data:image/jpeg;base64,BBBB"}}])
        out = mh.describe_image("")
        assert out.startswith("[IMAGE_DESCRIPTION_FAILED:"), out
    finally:
        mh._call_vision_model = orig
        mh.clear_pending()
        mh.set_pending_media(None)


def test_describe_never_raises_on_malformed_media():
    # A producer could set a malformed value directly; describe_image must still
    # return a marker rather than propagate an exception.
    mh._pending_media = 12345          # not a list/None
    mh._describe_media = 12345
    try:
        out = mh.describe_image("")
        assert out.startswith("[IMAGE_DESCRIPTION_FAILED:"), out
    finally:
        mh.clear_pending()
        mh.set_pending_media(None)


def test_sanitize_image_roundtrips_to_jpeg():
    from PIL import Image

    src = io.BytesIO()
    Image.new("RGB", (1, 1)).save(src, format="PNG")
    out = mh.sanitize_image(src.getvalue())
    assert out and isinstance(out, bytes)
    reopened = Image.open(io.BytesIO(out))
    assert reopened.format == "JPEG"


def test_extract_pdf_text_success_with_stubbed_pypdf():
    class FakePage:
        def extract_text(self):
            return "known page text"

    class FakePdfReader:
        def __init__(self, buf):
            self.pages = [FakePage()]

    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = FakePdfReader
    sys.modules["pypdf"] = fake_pypdf
    try:
        out = mh.extract_pdf_text(b"irrelevant bytes", "doc.pdf")
        assert "known page text" in out, out
        assert "[PDF: doc.pdf]" in out, out
    finally:
        del sys.modules["pypdf"]


def test_extract_pdf_text_failure_returns_marker_never_raises():
    # Deterministically exercise the failure path regardless of whether pypdf is
    # installed: stub PdfReader to raise. Must return a marker, never raise.
    class BoomReader:
        def __init__(self, buf):
            raise RuntimeError("corrupt pdf")

    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = BoomReader
    saved = sys.modules.get("pypdf")
    sys.modules["pypdf"] = fake_pypdf
    try:
        out = mh.extract_pdf_text(b"irrelevant bytes", "doc.pdf")
        assert out.startswith("[PDF: doc.pdf]"), out
        assert "Could not extract text" in out, out
    finally:
        if saved is not None:
            sys.modules["pypdf"] = saved
        else:
            del sys.modules["pypdf"]


def test_transcribe_audio_success_with_stubbed_openai_client():
    import openai

    class FakeTranscriptions:
        def create(self, **kwargs):
            assert kwargs["model"] == "openai/whisper-large-v3"
            return types.SimpleNamespace(text="hello world")

    class FakeAudio:
        transcriptions = FakeTranscriptions()

    class FakeClient:
        audio = FakeAudio()

    orig_openai_cls = openai.OpenAI
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    openai.OpenAI = lambda *a, **kw: FakeClient()
    try:
        out = mh.transcribe_audio(b"fake audio bytes", "voice.ogg")
        assert "hello world" in out, out
        assert "[AUDIO TRANSCRIPT: voice.ogg]" in out, out
    finally:
        openai.OpenAI = orig_openai_cls
        os.environ.pop("OPENROUTER_API_KEY", None)


def test_transcribe_audio_missing_key_returns_marker_never_raises():
    os.environ.pop("OPENROUTER_API_KEY", None)
    out = mh.transcribe_audio(b"fake audio bytes", "voice.ogg")
    assert out.startswith("[AUDIO TRANSCRIPT: voice.ogg]"), out


def test_generate_and_send_disabled():
    orig = mh._image_generation_allowed
    mh._image_generation_allowed = lambda: False
    try:
        out = mh.generate_and_send("a cat")
        assert out.startswith("IMAGE_DISABLED"), out
    finally:
        mh._image_generation_allowed = orig


def test_generate_and_send_unsafe():
    orig_allowed = mh._image_generation_allowed
    orig_unsafe = mh._prompt_is_unsafe
    mh._image_generation_allowed = lambda: True
    mh._prompt_is_unsafe = lambda prompt: True
    try:
        out = mh.generate_and_send("a cat")
        assert out.startswith("Refused"), out
    finally:
        mh._image_generation_allowed = orig_allowed
        mh._prompt_is_unsafe = orig_unsafe


def test_generate_and_send_generation_failure():
    orig_allowed = mh._image_generation_allowed
    orig_unsafe = mh._prompt_is_unsafe
    orig_gen = mh._generate_image_bytes
    mh._image_generation_allowed = lambda: True
    mh._prompt_is_unsafe = lambda prompt: False
    mh._generate_image_bytes = lambda prompt: None
    try:
        out = mh.generate_and_send("a cat")
        assert out.startswith("IMAGE_FAILED"), out
    finally:
        mh._image_generation_allowed = orig_allowed
        mh._prompt_is_unsafe = orig_unsafe
        mh._generate_image_bytes = orig_gen


def test_generate_and_send_success():
    orig_allowed = mh._image_generation_allowed
    orig_unsafe = mh._prompt_is_unsafe
    orig_gen = mh._generate_image_bytes
    orig_send, orig_chan = mh._live_send_photo, mh._live_channel

    sent = {}
    mh._image_generation_allowed = lambda: True
    mh._prompt_is_unsafe = lambda prompt: False
    mh._generate_image_bytes = lambda prompt: b"img"
    mh.register_channel(lambda image_bytes, caption=None: sent.update(
        bytes=image_bytes, caption=caption), lambda *a, **k: None, lambda *a, **k: None, object())
    try:
        out = mh.generate_and_send("a cat")
        assert out.startswith("IMAGE_SENT"), out
        assert sent["bytes"] == b"img", sent
    finally:
        mh._image_generation_allowed = orig_allowed
        mh._prompt_is_unsafe = orig_unsafe
        mh._generate_image_bytes = orig_gen
        mh._live_send_photo, mh._live_channel = orig_send, orig_chan


def test_generate_and_send_without_registered_channel():
    # Regression: the image generated but no channel was registered to send it.
    # Live runs hit this because the plugin loader execs plugin modules without
    # adding them to sys.modules, so looking the channel up by name found a
    # second, never-started copy.
    orig_allowed = mh._image_generation_allowed
    orig_unsafe = mh._prompt_is_unsafe
    orig_gen = mh._generate_image_bytes
    orig_send = mh._live_send_photo

    mh._image_generation_allowed = lambda: True
    mh._prompt_is_unsafe = lambda prompt: False
    mh._generate_image_bytes = lambda prompt: b"img"
    mh._live_send_photo = None
    try:
        out = mh.generate_and_send("a cat")
        assert out == "IMAGE_FAILED: generated but no channel is registered to send it", out
    finally:
        mh._image_generation_allowed = orig_allowed
        mh._prompt_is_unsafe = orig_unsafe
        mh._generate_image_bytes = orig_gen
        mh._live_send_photo = orig_send


def test_register_channel_wires_gate_and_sender():
    orig_send, orig_chan = mh._live_send_photo, mh._live_channel
    try:
        class Chan:
            reply_constraints = {"allow_image_generation": True}
        mh.register_channel(lambda *a, **k: None, lambda *a, **k: None, lambda *a, **k: None, Chan())
        assert mh._image_generation_allowed() is True
        Chan.reply_constraints = {"allow_image_generation": False}
        assert mh._image_generation_allowed() is False
    finally:
        mh._live_send_photo, mh._live_channel = orig_send, orig_chan


def test_generate_and_send_empty_prompt():
    out = mh.generate_and_send("   ")
    assert out == "IMAGE_FAILED: empty prompt", out


# --- speak skill tests -------------------------------------------------------

def test_speak_disabled():
    orig = mh._tts_allowed
    mh._tts_allowed = lambda: False
    try:
        out = mh.speak("hello")
        assert out.startswith("VOICE_DISABLED"), out
    finally:
        mh._tts_allowed = orig


def test_speak_synthesis_failure():
    orig_parts = mh.speech_parts
    mh.speech_parts = lambda text, voice: [(text, voice)]
    orig_allowed = mh._tts_allowed
    orig_unsafe = mh._prompt_is_unsafe
    orig_synth = mh._synthesise_speech
    mh._tts_allowed = lambda: True
    mh._prompt_is_unsafe = lambda text: False
    mh._synthesise_speech = lambda text, voice: None
    try:
        out = mh.speak("hello")
        assert out.startswith("VOICE_FAILED"), out
    finally:
        mh.speech_parts = orig_parts
        mh._tts_allowed = orig_allowed
        mh._prompt_is_unsafe = orig_unsafe
        mh._synthesise_speech = orig_synth


def test_speak_success():
    orig_parts = mh.speech_parts
    mh.speech_parts = lambda text, voice: [(text, voice)]
    orig_allowed = mh._tts_allowed
    orig_unsafe = mh._prompt_is_unsafe
    orig_synth = mh._synthesise_speech
    orig_send, orig_action, orig_chan = mh._live_send_voice, mh._live_send_chat_action, mh._live_channel

    actions = []
    sent = {}
    mh._tts_allowed = lambda: True
    mh._prompt_is_unsafe = lambda text: False
    mh._synthesise_speech = lambda text, voice: b"audio-bytes"
    mh._live_send_voice = lambda audio_bytes, caption=None: sent.update(bytes=audio_bytes)
    mh._live_send_chat_action = lambda action: actions.append(action)
    try:
        out = mh.speak("hello there")
        assert out == "VOICE_SENT", out
        assert sent["bytes"] == b"audio-bytes", sent
        assert "record_voice" in actions, actions
    finally:
        mh.speech_parts = orig_parts
        mh._tts_allowed = orig_allowed
        mh._prompt_is_unsafe = orig_unsafe
        mh._synthesise_speech = orig_synth
        mh._live_send_voice, mh._live_send_chat_action, mh._live_channel = orig_send, orig_action, orig_chan

def test_speak_succeeds_when_edge_voice_catalogue_fails():
    fake_edge = _fake_edge(None)
    fake_edge.list_voices = AsyncMock(
        side_effect=RuntimeError("Edge voice catalogue unavailable")
    )

    sent = {}
    actions = []

    with (
        patch.dict(sys.modules, {"edge_tts": fake_edge}),
        patch.object(sl, "_available_voices_cache", None),
        patch.object(sl, "_available_voices_retry_after", 0.0),
        patch.object(mh, "_tts_allowed", return_value=True),
        patch.object(mh, "_prompt_is_unsafe", return_value=False),
        patch.object(mh, "_synthesise_speech", return_value=b"audio-bytes"),
        patch.object(
            mh,
            "_live_send_voice",
            side_effect=lambda audio_bytes, caption=None: sent.update(
                bytes=audio_bytes
            ),
        ),
        patch.object(
            mh,
            "_live_send_chat_action",
            side_effect=lambda action: actions.append(action),
        ),
    ):
        out = mh.speak("hello there")

    assert out == "VOICE_SENT", out
    assert sent["bytes"] == b"audio-bytes", sent
    assert "record_voice" in actions, actions
    fake_edge.list_voices.assert_awaited_once()



def test_synthesise_speech_uses_edge_tts():
    calls = {}

    async def fake_stream():
        calls["called"] = True
        yield {"type": "audio", "data": b"mp3"}

    class FakeCommunicate:
        def __init__(self, text, voice):
            calls.update(text=text, voice=voice)

        def stream(self):
            return fake_stream()

    with patch.dict(sys.modules, {
        "edge_tts": _fake_edge(FakeCommunicate),
    }):
        result = mh._synthesise_speech("hello", "en-US-AriaNeural")

    assert result == b"mp3", result
    assert calls["called"] is True
    assert calls["text"] == "hello"
    assert calls["voice"] == "en-US-AriaNeural"


def test_synthesise_speech_falls_back_to_google_tts():
    class FailingCommunicate:
        def __init__(self, text, voice):
            pass

        def stream(self):
            async def fail():
                raise RuntimeError("edge down")
                yield None

            return fail()

    with patch.dict(sys.modules, {
        "edge_tts": _fake_edge(FailingCommunicate),
        "gtts": _fake_gtts(
            expected_text="hello",
            expected_lang="en",
        ),
    }):
        result = mh._synthesise_speech("hello", "en-US-AriaNeural")

    assert result == b"google-mp3", result


def test_synthesise_speech_falls_back_when_edge_returns_empty_audio():
    class EmptyCommunicate:
        def __init__(self, text, voice):
            pass

        def stream(self):
            async def empty():
                if False:
                    yield None

            return empty()

    with patch.dict(sys.modules, {
        "edge_tts": _fake_edge(EmptyCommunicate),
        "gtts": _fake_gtts(expected_lang="en"),
    }):
        result = mh._synthesise_speech("hello", "en-US-AriaNeural")

    assert result == b"google-mp3"


def test_synthesise_speech_falls_back_when_edge_times_out():
    import asyncio

    class SlowCommunicate:
        def __init__(self, text, voice):
            pass

        async def stream(self):
            await asyncio.sleep(1)
            yield {"type": "audio", "data": b"edge-mp3"}

    with (
        patch.dict(sys.modules, {
            "edge_tts": _fake_edge(SlowCommunicate),
            "gtts": _fake_gtts(expected_lang="en"),
        }),
        patch.object(mh, "EDGE_TTS_TIMEOUT", 0.01),
    ):
        result = mh._synthesise_speech("hello", "en-US-AriaNeural")

    assert result == b"google-mp3", result


def test_google_language_mapping_matches_gtts_codes():
    expected = {
        "nb": "no",
        "nn": "no",
        "he": "iw",
        "fil": "tl",
        "en": "en",
    }

    for language, google_code in expected.items():
        assert sl.google_tts_language(language) == google_code


def test_synthesise_speech_maps_detected_norwegian_to_google_code():
    class FailingCommunicate:
        def __init__(self, text, voice):
            pass

        def stream(self):
            async def fail():
                raise RuntimeError("edge unavailable")
                yield None

            return fail()

    with (
        patch.dict(sys.modules, {
            "edge_tts": _fake_edge(FailingCommunicate),
            "gtts": _fake_gtts(
                expected_text="Hei",
                expected_lang="no",
            ),
        }),
        patch.object(mh, "detect_language", return_value="nb"),
    ):
        result = mh._synthesise_speech("Hei", "en-US-AriaNeural")

    assert result == b"google-mp3", result


def test_google_tts_verifies_https_including_the_english_retry():
    import requests

    for language in ("en", "xx"):
        response = requests.Response()
        response.status_code = 200
        response.raw = io.BytesIO(b'["jQ1olc","[\\"bXAz\\"]"]')
        response.request = requests.Request("POST", "https://translate.google.com/").prepare()
        with (
            patch.object(mh, "detect_language", return_value=language),
            patch.object(requests.Session, "send", return_value=response) as send,
        ):
            result = mh._synthesise_google("Hello", "en-US-AriaNeural")

        assert result == b"mp3", result
        send.assert_called_once()
        assert send.call_args.kwargs["verify"] is True
        assert send.call_args.kwargs["timeout"] == mh.GOOGLE_TTS_TIMEOUT


def test_google_tts_returns_none_on_network_errors():
    import requests

    for error in (
        requests.Timeout("Google TTS request timed out"),
        requests.exceptions.SSLError("invalid certificate"),
    ):
        with (
            patch.object(mh, "detect_language", return_value="en"),
            patch.object(requests.Session, "send", side_effect=error) as send,
        ):
            result = mh._synthesise_google("Hello", "en-US-AriaNeural")

        assert result is None
        send.assert_called_once()
        assert send.call_args.kwargs["verify"] is True
        assert send.call_args.kwargs["timeout"] == mh.GOOGLE_TTS_TIMEOUT
        assert all(timeout > 0 for timeout in send.call_args.kwargs["timeout"])


if __name__ == "__main__":
    test_no_image_returns_marker()
    test_describe_memoizes_per_turn()
    test_describe_never_raises()
    test_describe_never_raises_on_malformed_media()
    test_sanitize_image_roundtrips_to_jpeg()
    test_extract_pdf_text_success_with_stubbed_pypdf()
    test_extract_pdf_text_failure_returns_marker_never_raises()
    test_transcribe_audio_success_with_stubbed_openai_client()
    test_transcribe_audio_missing_key_returns_marker_never_raises()
    test_generate_and_send_disabled()
    test_generate_and_send_unsafe()
    test_generate_and_send_generation_failure()
    test_generate_and_send_success()
    test_generate_and_send_without_registered_channel()
    test_register_channel_wires_gate_and_sender()
    test_generate_and_send_empty_prompt()
    test_speak_disabled()
    test_speak_synthesis_failure()
    test_speak_success()
    test_speak_succeeds_when_edge_voice_catalogue_fails()
    test_synthesise_speech_uses_edge_tts()
    test_synthesise_speech_falls_back_to_google_tts()
    test_synthesise_speech_falls_back_when_edge_returns_empty_audio() 
    test_synthesise_speech_falls_back_when_edge_times_out()
    test_google_language_mapping_matches_gtts_codes()
    test_synthesise_speech_maps_detected_norwegian_to_google_code()
    test_google_tts_verifies_https_including_the_english_retry()
    test_google_tts_returns_none_on_network_errors()
    print("all media_handler tests passed")
