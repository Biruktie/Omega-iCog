import base64
import hashlib
import threading
import time
import logging
import sys
from contextlib import contextmanager
from collections import OrderedDict
from itertools import groupby
from speech_text import prepare_speech
from speech_language import speech_parts, speech_chunks, detect_language, google_tts_language

logger = logging.getLogger(__name__)

# skills.metta reaches this via `py-call (media_handler.describe_image ...)`, so
# the module must resolve under the bare name `media_handler` even when Python
# first imported it under a package-qualified name. Alias to one instance.
_self = sys.modules[__name__]
sys.modules.setdefault("media_handler", _self)
sys.modules.setdefault("src.media_handler", _self)

_lock = threading.Lock()
_pending_media = None
_pending_context = None
_describe_media = None          # kept for describe-image; survives a reply's clear_pending()
_pending_description = {}       # (image_key, query) -> caption, per-turn memo

VISION_PROMPT = (
    "Describe this image for a text-only assistant. Report objects, any visible "
    "text verbatim, layout, and notable details. Be concise and factual."
)


def set_pending_media(media):
    global _pending_media, _describe_media
    with _lock:
        _pending_media = media
        imgs = _image_parts(media)
        if imgs:
            _describe_media = media
            _pending_description.clear()
            logger.info("[IMGDBG] pending media set: %d image part(s)", len(imgs))
        elif media is None:
            _describe_media = None


def get_pending_media():
    with _lock:
        return _pending_media


def set_pending_context(text):
    global _pending_context
    with _lock:
        _pending_context = text


def get_pending_context():
    with _lock:
        return _pending_context


def clear_pending():
    """Drop the pending slots and memo once the agent has replied. The describe
    source is kept so describe-image is not blinded mid-turn by a reply."""
    global _pending_media, _pending_context
    with _lock:
        had = bool(_pending_media) or bool(_pending_context)
        _pending_media = None
        _pending_context = None
        _pending_description.clear()
    if had:
        logger.info("[IMGDBG] clear_pending: dropped pending media/context (describe source kept)")


def _image_parts(media):
    return [p for p in (media or [])
            if isinstance(p, dict) and p.get("type") == "image_url"]


def _image_key(image_parts):
    urls = "".join((p.get("image_url") or {}).get("url", "") for p in image_parts)
    return hashlib.sha256(urls.encode("utf-8")).hexdigest()


def _call_vision_model(image_parts, prompt):
    """Vision call via the configured vision provider. Isolated so tests stub it."""
    from vision import vision_chat
    return vision_chat(image_parts, prompt)


def describe_image(query=""):
    """describe-image skill: caption the currently-pending image so a non-vision
    agent can 'see' it. Memoized per turn; never raises — any failure, including
    a malformed pending-media value from a producer, returns a failure marker."""
    try:
        query = (query or "").strip()
        with _lock:
            source = _pending_media if _image_parts(_pending_media) else _describe_media
        parts = _image_parts(source)
        if not parts:
            logger.info("[IMGDBG] describe_image: no pending image available")
            return "[NO_IMAGE: nothing is attached to describe]"

        key = (_image_key(parts), query)
        with _lock:
            cached = _pending_description.get(key)
        if cached is not None:
            return cached

        prompt = VISION_PROMPT if not query else f"{VISION_PROMPT} Focus on: {query}"
        caption = _call_vision_model(parts, prompt)
        result = f"[IMAGE DESCRIPTION]\n{caption}"
        with _lock:
            _pending_description[key] = result
        logger.info("Described pending image: %d chars", len(caption))
        return result
    except Exception as e:
        logger.error("Image description failed: %s", e)
        return f"[IMAGE_DESCRIPTION_FAILED: {e}]"


def image_to_data_uri(file_bytes, mime_type):
    """Helper for image ingestion (used by the channel plugin)."""
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_content(text, media):
    return [{"type": "text", "text": text}] + media


def sanitize_image(file_bytes, max_dim=2048, quality=85):
    """Re-encode an image via Pillow: strips EXIF/metadata and destroys most
    LSB-embedded steganographic payloads. Raises on undecodable input so the
    caller's existing error path rejects the file."""
    from PIL import Image
    from io import BytesIO
    img = Image.open(BytesIO(file_bytes))
    img = img.convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    out = BytesIO()
    img.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def extract_pdf_text(file_bytes, filename, max_chars=20000):
    try:
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[truncated]"
        logger.info(f"Extracted text from {filename}: {len(text)}")
        return f"[PDF: {filename}]\n{text}"
    except Exception as e:
        logger.error(f"PDF extraction failed for {filename}: {e}")
        return f"[PDF: {filename}]\n[Could not extract text: {e}]"


def transcribe_audio(file_bytes, filename, model="openai/whisper-large-v3", max_bytes=25 * 1024 * 1024, language=None, temperature=None):
    """Transcribe an audio/voice file to text via OpenRouter Whisper Large V3.

    Returns a marker-prefixed string for the PDF-style context slot, or an
    error note (never raises) so the agent still gets a usable turn.
    """
    try:
        if len(file_bytes) > max_bytes:
            return f"[AUDIO TRANSCRIPT: {filename}]\n[Audio too large to transcribe]"

        import io, openai
        import gateway

        base_url, api_key = gateway.upstream(
            "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
        if not api_key:
            return f"[AUDIO TRANSCRIPT: {filename}]\n[Could not transcribe: OPENROUTER_API_KEY is not set]"

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        audio_file = io.BytesIO(file_bytes)
        audio_file.name = filename
        kwargs = { "model": model, "file": audio_file,}
        if language:
            kwargs["language"] = language

        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = client.audio.transcriptions.create(**kwargs)
        text = (getattr(resp, "text", "") or "").strip()
        logger.info(f"Transcribed {filename}: {len(text)} chars")

        return f"[AUDIO TRANSCRIPT: {filename}]\n{text}"

    except Exception as e:
        logger.error(f"Audio transcription failed for {filename}: {e}")
        return f"[AUDIO TRANSCRIPT: {filename}]\n[Could not transcribe: {e}]"


# --- Image generation (outbound) -------------------------------------------
# Provider-agnostic text-to-image. The default is OpenRouter FLUX; the image
# provider is chosen INDEPENDENTLY of the chat `provider` (not every chat
# provider can generate images) via IMAGE_PROVIDER / IMAGE_MODEL env vars.
# OpenRouter's image endpoint is NOT OpenAI-SDK compatible (dedicated
# POST /api/v1/images), so that path uses a raw requests POST; both styles
# return the image as base64 in data[0].b64_json.
IMAGE_PROVIDERS = {
    "OpenRouter": {
        "style": "openrouter",
        "route": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "path": "images",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "black-forest-labs/flux.2-pro",
    },
    "OpenAI": {
        "style": "openai_sdk",
        "route": "openai",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-image-1",
    },
}


def _generate_image_bytes(prompt):
    """Generate an image for `prompt` via the configured image provider.
    Returns raw image bytes, or None on any failure (never raises)."""
    import os
    import gateway
    provider_name = os.environ.get("IMAGE_PROVIDER", "OpenRouter")
    cfg = IMAGE_PROVIDERS.get(provider_name) or IMAGE_PROVIDERS["OpenRouter"]
    model = os.environ.get("IMAGE_MODEL", cfg["default_model"])
    base_url, api_key = gateway.upstream(cfg["route"], cfg["base_url"], cfg["key_env"])
    if not api_key:
        logger.error(f"Image generation: {cfg['key_env']} not set for provider {provider_name}")
        return None
    try:
        if cfg["style"] == "openrouter":
            import requests
            resp = requests.post(
                f"{base_url.rstrip('/')}/{cfg['path']}",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": model, "prompt": prompt},
                timeout=120,
            )
            resp.raise_for_status()
            b64 = resp.json()["data"][0]["b64_json"]
        elif cfg["style"] == "openai_sdk":
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            resp = client.images.generate(model=model, prompt=prompt)
            b64 = resp.data[0].b64_json
        else:
            logger.error(f"Image generation: unknown style for provider {provider_name}")
            return None
        image_bytes = base64.b64decode(b64)
        logger.info(f"Generated image via {provider_name}/{model}: {len(image_bytes)} bytes")
        return image_bytes
    except Exception as e:
        logger.error(f"Image generation failed ({provider_name}/{model}): {e}")
        return None


_live_send_photo = None
_live_send_voice = None
_live_send_chat_action = None
_live_channel = None


def register_channel(send_photo, send_voice, send_chat_action, channel):
    """Give this module a direct handle on the LIVE channel, called by the
    channel plugin's loadOmegaPlugin.

    Registration is explicit rather than looked up by module name because the
    plugin loader execs plugin modules without adding them to sys.modules, so
    `import telegram` here would build a second, never-started copy whose
    bot and event loop are None — generated images would be produced and then
    dropped."""
    global _live_send_photo, _live_send_voice, _live_send_chat_action, _live_channel
    _live_send_photo = send_photo
    _live_send_voice = send_voice
    _live_send_chat_action = send_chat_action
    _live_channel = channel


def _image_generation_allowed():
    """Read the allow_image_generation gate from the active channel's reply
    constraints. Fail closed (False) if unavailable."""
    try:
        constraints = getattr(_live_channel, "reply_constraints", None) or {}
        return bool(constraints.get("allow_image_generation", False))
    except Exception as e:
        logger.error(f"Could not read allow_image_generation gate: {e}")
        return False


def _prompt_is_unsafe(prompt):
    """Run the ethics classifier on the image prompt before spending money on
    generation. Sync bridge over the async is_category_blocked, mirroring
    tg_channel.send_message. Fail closed (unsafe) on unexpected error."""
    import asyncio
    from config_helper import is_category_blocked
    try:
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(is_category_blocked(prompt))
        except RuntimeError:
            return asyncio.run(is_category_blocked(prompt))
    except Exception as e:
        logger.error(f"Image prompt ethics check failed: {e}")
        return True


def generate_and_send(prompt):
    """generate-image skill: generate an image for `prompt` and send it to the
    user. Returns a short status string (never raises) that the agent sees next
    cycle in LAST_SKILL_USE_RESULTS. Bytes are generated and dispatched here
    (not through MeTTa, which only handles strings)."""
    prompt = (prompt or "").strip()
    if not prompt:
        return "IMAGE_FAILED: empty prompt"
    if not _image_generation_allowed():
        return "IMAGE_DISABLED: image generation is turned off"
    if _prompt_is_unsafe(prompt):
        return "Refused: unsafe image prompt"
    image_bytes = _generate_image_bytes(prompt)
    if not image_bytes:
        return f"IMAGE_FAILED: could not generate image for: {prompt}"
    if _live_send_photo is None:
        return "IMAGE_FAILED: generated but no channel is registered to send it"
    try:
        _live_send_photo(image_bytes, caption=prompt[:1024])
    except Exception as e:
        logger.error(f"Failed to send generated image: {e}")
        return f"IMAGE_FAILED: generated but could not send: {e}"
    return f"IMAGE_SENT: {prompt}"

# --- Text-to-speech (outbound) ---------------------------------------------
# speak skill: synthesise a voice message and send it via sendVoice.
# Uses Edge TTS first, with automatic Google TTS fallback if Edge fails.
# Voice is configurable via EDGE_TTS_VOICE.

DEFAULT_TTS_VOICE = "en-US-AriaNeural"
EDGE_TTS_TIMEOUT = 30
GOOGLE_TTS_TIMEOUT = (5, 15)
RECORDING_REFRESH_SECONDS = 4
MAX_VOICE_REQUESTS = 100
_voice_requests = OrderedDict()
_voice_delivery_lock = threading.Lock()
_turn = 0


def next_turn():
    global _turn
    _turn += 1


class SpeechDelivery:
    """Voice-only retry state; forgotten after restart or cache eviction."""

    def __init__(self, request_id):
        self.request_id = request_id
        with _voice_delivery_lock:
            self._parts = _voice_requests.setdefault(request_id, {})
            _voice_requests.move_to_end(request_id)
            while len(_voice_requests) > MAX_VOICE_REQUESTS:
                _voice_requests.popitem(last=False)

    def _touch(self):
        # Retain an active instance's state after eviction without reviving it
        # or touching a newer entry created with the same request ID.
        if _voice_requests.get(self.request_id) is self._parts:
            _voice_requests.move_to_end(self.request_id)

    def get(self, part):
        with _voice_delivery_lock:
            self._touch()
            return self._parts.get(part, {}).get("state", "pending")

    def set(self, part, state, message_id=None):
        self.set_many([part], state, message_id)

    def set_many(self, parts, state, message_id=None):
        """Update all segments in one voice upload under the same lock."""
        updates = {part: {"state": state, "message_id": message_id} for part in parts}
        with _voice_delivery_lock:
            self._touch()
            self._parts.update(updates)

    def attempt(self):
        with _voice_delivery_lock:
            self._touch()
            return self._parts.get("attempt")

    def record_attempt(self, turn, complete, copies):
        with _voice_delivery_lock:
            self._touch()
            self._parts["attempt"] = {"turn": turn, "complete": complete, "copies": copies}

    def reset(self):
        with _voice_delivery_lock:
            self._touch()
            self._parts.clear()


@contextmanager
def _recording_indicator():
    """Refresh the recording action for this speech operation only."""
    send_action = _live_send_chat_action
    if send_action is None:
        yield
        return
    channel = _live_channel
    chat_id = getattr(channel, "chat_id", None)
    stop = threading.Event()

    def refresh():
        # Do not move a pending recording indicator to another conversation.
        if channel is not None and getattr(channel, "chat_id", None) != chat_id:
            return
        try:
            send_action("record_voice")
        except Exception as exc:
            logger.warning(f"Could not send record_voice chat action: {exc}")

    def run():
        next_refresh = started + RECORDING_REFRESH_SECONDS
        while not stop.wait(max(0, next_refresh - time.monotonic())):
            next_refresh = time.monotonic() + RECORDING_REFRESH_SECONDS
            refresh()

    started = time.monotonic()
    refresh()
    worker = threading.Thread(target=run, name="telegram-recording", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def _tts_allowed():
    """Read the allow_voice_reply gate from the active channel's reply
    constraints. Fail closed (False) if unavailable."""
    try:
        constraints = getattr(_live_channel, "reply_constraints", None) or {}
        return bool(constraints.get("allow_voice_reply", False))
    except Exception as e:
        logger.error(f"Could not read allow_voice_reply gate: {e}")
        return False



def _synthesise_edge(chunk_text, chunk_voice):
    """Synthesize speech with Edge TTS, enforcing a timeout on streaming."""
    import asyncio
    import concurrent.futures
    import io
    import edge_tts

    async def _stream():
        with io.BytesIO() as buf:
            async for chunk in edge_tts.Communicate(
                chunk_text, voice=chunk_voice
            ).stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

    async def _run():
        return await asyncio.wait_for(
            _stream(),
            timeout=EDGE_TTS_TIMEOUT,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    # asyncio.run() cannot execute inside an already-running event loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result()


def _google_lang_from_voice(chunk_text, chunk_voice):
    language = detect_language(chunk_text)

    if language:
        language = language.lower()
    elif isinstance(chunk_voice, str) and chunk_voice.strip():
        language = chunk_voice.split("-", 1)[0].lower()
    else:
        language = "en"

    return google_tts_language(language)


def _synthesise_google(chunk_text, chunk_voice):
    """Synthesize speech using Google TTS over verified HTTPS."""
    import io
    import re
    import urllib.request
    import requests
    from gtts import gTTS

    class VerifiedGTTS(gTTS):
        # gTTS has no verification option; reuse its prepared requests, not its transport.
        def stream(self):
            with requests.Session() as session:
                for request in self._prepare_requests():
                    response = session.send(
                        request=request,
                        verify=True,
                        proxies=urllib.request.getproxies(),
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    for line in response.iter_lines(chunk_size=1024):
                        line = line.decode("utf-8")
                        if "jQ1olc" not in line:
                            continue
                        audio = re.search(r'jQ1olc","\[\\"(.*)\\"]', line)
                        if audio is None:
                            raise RuntimeError("Google TTS returned invalid audio data")
                        yield base64.b64decode(audio.group(1))

    language = _google_lang_from_voice(chunk_text, chunk_voice)

    try:
        with io.BytesIO() as buf:
            VerifiedGTTS(
                text=chunk_text,
                lang=language,
                timeout=GOOGLE_TTS_TIMEOUT
            ).write_to_fp(buf)
            return buf.getvalue()

    except ValueError:
        if language == "en":
            logger.warning(
                "Google TTS does not support language %s", language
            )
            return None

        logger.warning(
            "Google TTS does not support language %s; falling back to English",
            language,
        )

    except Exception as exc:
        logger.warning("Google TTS synthesis failed: %s", exc)
        return None

    try:
        with io.BytesIO() as buf:
            VerifiedGTTS(
                text=chunk_text,
                lang="en",
                timeout=GOOGLE_TTS_TIMEOUT
            ).write_to_fp(buf)
            return buf.getvalue()
    except Exception as exc:
        logger.warning("Google TTS synthesis failed: %s", exc)
        return None



def _synthesise_speech(text, voice):
    """Synthesize MP3 audio using Edge TTS, falling back to Google TTS."""

    # Always try Edge first.
    try:
        audio = _synthesise_edge(text, voice)
        if audio:
            return audio

        logger.warning("Edge TTS returned empty audio; trying Google TTS")
    except Exception as exc:
        logger.warning("Edge TTS failed: %s; trying Google TTS", exc)

    # Automatically fall back to Google if Edge fails.
    try:
        audio = _synthesise_google(text, voice)
        if audio:
            return audio
    except Exception as exc:
        logger.warning("Google TTS failed: %s", exc)

    return None



def speak(text=""):
    """speak skill: synthesise `text` as a voice message and send it to the
    user via Telegram sendVoice. Returns a short status string (never raises)."""
    from config import config_get_by_key
    # Match send_message: decode escaped newlines before splitting.
    if not isinstance(text, str):
        return "VOICE_INVALID_INPUT: provide text to speak; do not retry unchanged input"
    text = text.replace("\\n", "\n").strip()
    if not text:
        return "VOICE_INVALID_INPUT: empty text; ask for text instead of retrying"
    if not _tts_allowed():
        return "VOICE_DISABLED: voice replies are turned off"
    if _prompt_is_unsafe(text):
        return "Refused: unsafe voice content"
    text = prepare_speech(text)
    if not text:
        return "VOICE_INVALID_INPUT: no speakable text after cleanup; ask for text instead of retrying"
    voice = config_get_by_key("EDGE_TTS_VOICE", DEFAULT_TTS_VOICE)
    if _live_send_voice is None:
        return "VOICE_FAILED: no channel is registered to send it"
    target = (_live_channel, getattr(_live_channel, "chat_id", None),
              getattr(_live_channel, "_reply_to_id", None))
    turn = _turn
    copies = 0
    ledger = None
    if target[1] is not None and target[2] is not None:
        request_id = hashlib.sha256(repr((target[1:], text, voice)).encode()).hexdigest()
        try:
            ledger = SpeechDelivery(request_id)
        except Exception:
            logger.exception("Could not initialize voice retry tracking")
            return "VOICE_FAILED: delivery checkpoint unavailable; no audio sent"
        attempt = ledger.attempt()
        copies = attempt["copies"] if attempt else 0
        if attempt and attempt["turn"] == turn:
            ledger.reset()
        elif attempt and attempt["complete"]:
            times = "once" if copies == 1 else f"{copies} times"
            return f"VOICE_DUPLICATE: already delivered {times} for this message; not sent again"
    failures = []
    sent = 0

    def synthesise(chunk):
        audio = b""
        for chunk_voice, run in groupby(chunk, key=lambda part: part[2]):
            run_text = "".join(piece for _, piece, _ in run)
            try:
                data = _synthesise_speech(run_text, chunk_voice) if chunk_voice else None
            except Exception:
                logger.exception("Speech chunk synthesis failed")
                data = None
            if not data:
                return None
            audio += data
        return audio

    def fail(part):
        failures.append((part[0], part[1], "synthesis failed"))
        if ledger:
            ledger.set(part[0], "failed")

    def deliver(chunk):
        audio = synthesise(chunk) or synthesise(chunk)
        if audio:
            upload(chunk, audio)
            return
        if len(chunk) == 1:
            fail(chunk[0])
            return
        group, group_audio = [], b""
        for part in chunk:
            audio = synthesise([part])
            if audio:
                group.append(part)
                group_audio += audio
                continue
            if group:
                upload(group, group_audio)
                group, group_audio = [], b""
            fail(part)
        if group:
            upload(group, group_audio)

    def upload(chunk, audio):
        nonlocal sent
        indices = [i for i, _, _ in chunk]
        if (getattr(_live_channel, "chat_id", None),
                getattr(_live_channel, "_reply_to_id", None)) != target[1:]:
            failures.extend((i, piece, "conversation changed") for i, piece, _ in chunk)
            return
        if ledger:
            ledger.set_many(indices, "uploading")
        try:
            message_id = _live_send_voice(audio)
            if ledger:
                ledger.set_many(indices, "sent", message_id)
            sent += len(indices)
        except Exception:
            logger.exception("Voice chunk delivery uncertain")
            if ledger:
                ledger.set_many(indices, "uncertain")
            failures.extend((i, piece, "delivery uncertain; do not resend") for i, piece, _ in chunk)

    with _recording_indicator():
        try:
            pieces = speech_parts(text, voice)
        except Exception as exc:
            logger.exception("Could not select speech voices")
            return f"VOICE_FAILED: could not select a language-compatible voice: {exc}"
        pending = []
        for index, (piece, voice) in enumerate(pieces):
            state = ledger.get(index) if ledger else "pending"
            if state == "sent":
                sent += 1
                continue
            if state in ("uploading", "uncertain"):
                failures.append((index, piece, "delivery uncertain; do not resend"))
                continue
            pending.append((index, piece, voice))
        for chunk in speech_chunks(pending):
            deliver(chunk)
    if ledger:
        ledger.record_attempt(turn, not failures, copies if failures else copies + 1)
    if failures:
        details = "; ".join(f"part {i + 1}/{len(pieces)}: {reason} ({piece[:80]})"
                            for i, piece, reason in failures)
        if (_live_channel is not None and getattr(_live_channel, "chat_id", None) == target[1]
                and (ledger is None or ledger.get(-1) == "pending")):
            try:
                if ledger:
                    ledger.set(-1, "uploading")
                _live_channel.send_message("Some speech could not be delivered: " + details[:3000],
                                           chat_id=target[1], reply_to_id=target[2])
                if ledger:
                    ledger.set(-1, "sent")
            except Exception:
                logger.exception("Could not deliver speech failure notice")
        return f"{'VOICE_PARTIAL' if sent else 'VOICE_FAILED'}: {sent} parts already sent; {details}"
    return "VOICE_SENT"
