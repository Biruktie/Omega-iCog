"""Detect speech language locally and select voices without changing config."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
import re
import threading
from time import monotonic

from text_splitter import split_for_telegram


def speech_chunks(parts):
    """Pack indexed sentence pieces without crossing the text limit."""
    chunk = []
    size = 0
    for part in parts:
        _, text, _ = part
        if chunk and size + len(text) > 4096:
            yield chunk
            chunk, size = [], 0
        chunk.append(part)
        size += len(text)
    if chunk:
        yield chunk

logger = logging.getLogger(__name__)
DEFAULT_VOICE = "en-US-AriaNeural"
PREFERRED_VOICES = {"en": DEFAULT_VOICE, "ru": "ru-RU-SvetlanaNeural"}
PREFERRED_LOCALES = {"en": "en-US", "pt": "pt-BR", "ar": "ar-SA", "ur": "ur-PK"}
_SCRIPTS = {
    "cyrillic": r"[\u0400-\u052f]",
    "kana": r"[\u3040-\u30ff]",
    "hangul": r"[\uac00-\ud7af]",
    "han": r"[\u3400-\u9fff]",
    "arabic": r"[\u0600-\u06ff]",
    "hebrew": r"[\u0590-\u05ff]",
    "greek": r"[\u0370-\u03ff]",
    "devanagari": r"[\u0900-\u097f]",
    "thai": r"[\u0e00-\u0e7f]",
}
_IDEOGRAPHIC_SCRIPTS = {"kana", "hangul", "han"}
_SCRIPT_LANGUAGES = {"kana": "ja", "hangul": "ko", "han": "zh", "arabic": "ar",
                     "hebrew": "he", "greek": "el", "devanagari": "hi", "thai": "th"}
_VOICE_SCRIPTS = {
    **dict.fromkeys(("ru", "uk", "bg", "be", "mk", "sr", "kk", "mn"), {"cyrillic"}),
    **dict.fromkeys(("ar", "fa", "ur", "ps"), {"arabic"}),
    **dict.fromkeys(("hi", "mr", "ne"), {"devanagari"}),
    "zh": {"han"}, "ja": {"kana", "han"}, "ko": {"hangul", "han"},
    "he": {"hebrew"}, "el": {"greek"}, "th": {"thai"},
}


@lru_cache(maxsize=1)
def _detector():
    from lingua import LanguageDetectorBuilder
    # Load models lazily; standard mode also recognises short Cyrillic phrases.
    return LanguageDetectorBuilder.from_all_languages().build()


@lru_cache(maxsize=1)
def _cyrillic_detector():
    from lingua import LanguageDetectorBuilder, Language
    return LanguageDetectorBuilder.from_languages(Language.RUSSIAN, Language.UKRAINIAN).build()


def _script_is_main(text, pattern):
    words = len(re.findall(pattern, text))
    return words > 0 and words >= len(re.findall(r"[A-Za-z]+", text))


def detect_language(text):
    # Script hints protect short phrases and non-Latin text containing product
    # names from being routed to an English-only voice.
    if _script_is_main(text, r"[\u3040-\u30ff]+"):
        return "ja"
    if _script_is_main(text, r"[\uac00-\ud7af]+"):
        return "ko"
    if _script_is_main(text, r"[\u3400-\u9fff]+"):
        return "zh"
    if _script_is_main(text, r"[\u0400-\u052f]+"):
        if re.search(r"[іїєґІЇЄҐ]", text):
            return "uk"
        text = re.sub(r"[A-Za-z]+", " ", text)
        if re.search(r"[ыэёъЫЭЁЪ]", text):
            return "ru"
    else:
        # A brief foreign quotation must not outweigh a mostly Latin reply.
        non_latin = r"[^\W\d_A-Za-z\u00c0-\u024f]+"
        if len(re.findall(r"[A-Za-z\u00c0-\u024f]+", text)) > len(re.findall(non_latin, text)):
            text = re.sub(non_latin, " ", text)
    letters = [char.lower() for char in text if char.isalpha()]
    if len(letters) < 5 or len(set(letters)) < 3:
        return None
    scores = _detector().compute_language_confidence_values(text)
    
    if _script_is_main(text, r"[\u0400-\u052f]+") and (
            not scores or scores[0].value < .35):
        scores = _cyrillic_detector().compute_language_confidence_values(text)
        if scores and scores[0].value >= .60:
            return scores[0].language.iso_code_639_1.name.lower()
        return None
    # Scores are relative across all languages, not calibrated probabilities.
    if not scores or scores[0].value < .35:
        return None
    if len(scores) > 1 and scores[0].value - scores[1].value < .10:
        return None
    return scores[0].language.iso_code_639_1.name.lower()


def _language(code):
    code = code.lower().split("-")[0]
    return {"nb": "no", "nn": "no"}.get(code, code)


def google_tts_language(code):
    """Convert a normalized language code to a Google TTS language code."""
    code = _language(code)
    return {"he": "iw", "fil": "tl"}.get(code, code)


def _present(text, script):
    pattern = _SCRIPTS[script]
    return re.search(pattern if script in _IDEOGRAPHIC_SCRIPTS else pattern + "{2,}", text)


def _count(text, script):
    return len(re.findall(_SCRIPTS[script], text))


def _cyrillic_language(text):
    text = " ".join(re.findall(r"[\u0400-\u052f]+", text))
    if re.search(r"[іїєґІЇЄҐ]", text):
        return "uk"
    if re.search(r"[ыэёъЫЭЁЪ]", text):
        return "ru"
    scores = _cyrillic_detector().compute_language_confidence_values(text)
    if scores and scores[0].value >= .60:
        return scores[0].language.iso_code_639_1.name.lower()
    return "ru"


def _switch_language(text, voice):
    own = _VOICE_SCRIPTS.get(_language(voice), set())
    unreadable = [script for script in _SCRIPTS if script not in own and _present(text, script)]
    if not unreadable:
        return None
    script = next((name for name in ("kana", "hangul") if name in unreadable),
                  max(unreadable, key=lambda name: _count(text, name)))
    if _count(text, script) <= sum(_count(text, name) for name in own):
        return None
    if script == "cyrillic":
        return _cyrillic_language(text)
    return _SCRIPT_LANGUAGES[script]


def _segments(text):
    segments = []
    start = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+|[。！？]+[」』）〉》】”’)\]]*\s*|\n+", text):
        if boundary.end() > start:
            segments.append(text[start:boundary.end()])
            start = boundary.end()
    if start < len(text):
        segments.append(text[start:])
    return segments


_available_voices_cache = None
_available_voices_lock = threading.Lock()
_available_voices_retry_after = 0.0

VOICE_CATALOGUE_RETRY_COOLDOWN = 30


def available_voices():
    """Load Edge's voice catalogue, caching successes only for retries after outages."""
    global _available_voices_cache, _available_voices_retry_after

    with _available_voices_lock:
        if _available_voices_cache is not None:
            return _available_voices_cache

        if monotonic() < _available_voices_retry_after:
            return []

        try:
            import edge_tts

            async def fetch():
                return await asyncio.wait_for(edge_tts.list_voices(), timeout=10)

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                voices = asyncio.run(fetch())
            else:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    voices = pool.submit(asyncio.run, fetch()).result()

            if voices:
                _available_voices_cache = voices
                _available_voices_retry_after = 0.0
            else:
                _available_voices_retry_after = monotonic() + VOICE_CATALOGUE_RETRY_COOLDOWN

            return voices or []

        except Exception as exc:
            _available_voices_retry_after = monotonic() + VOICE_CATALOGUE_RETRY_COOLDOWN
            
            logger.warning(
                "Could not load Edge TTS voice catalogue: %s", exc
            )
            return []


def _clear_available_voices_cache():
    global _available_voices_cache, _available_voices_retry_after

    with _available_voices_lock:
        _available_voices_cache = None
        _available_voices_retry_after = 0.0


available_voices.cache_clear = _clear_available_voices_cache


def select_voice(language, configured_voice):
    # Keep the configured voice for its own language and uncertain text.
    if language is None or _language(language) == _language(configured_voice):
        return configured_voice

    language = _language(language)
    voices = available_voices()

    if not voices:
        logger.warning(
            "Edge TTS voice catalogue unavailable; using configured voice %s",
            configured_voice,
        )
        return configured_voice

    gender = next(
        (
            voice.get("Gender")
            for voice in voices
            if voice.get("ShortName") == configured_voice
        ),
        None,
    )

    if gender not in ("Female", "Male"):
        raise ValueError(
            f"Cannot determine gender of configured voice: {configured_voice!r}"
        )

    candidates = sorted(
        voice["ShortName"]
        for voice in voices
        if _language(voice.get("Locale", "")) == language
        and voice.get("ShortName")
        and voice.get("Gender") == gender
    )

    if not candidates:
        logger.warning(
            "No %s speech voice for detected language %s; using configured voice %s",
            gender.lower(),
            language,
            configured_voice,
        )
        return configured_voice

    preferred = PREFERRED_VOICES.get(language)
    if preferred in candidates:
        return preferred

    locale = PREFERRED_LOCALES.get(language, f"{language}-{language.upper()}")
    return next(
        (voice for voice in candidates if voice.startswith(locale + "-")),
        candidates[0],
    )


def validated_voice(configured_voice):
    """Validate the configured voice against Edge's cached voice catalogue."""
    voices = available_voices()

    # Do not reject a configured voice when the catalogue is unavailable.
    if not voices:
        logger.warning(
            "Edge TTS voice catalogue unavailable; retaining configured voice %r",
            configured_voice,
        )
        return configured_voice

    if any(
        voice.get("ShortName") == configured_voice
        for voice in voices
    ):
        return configured_voice

    logger.warning(
        "Unknown configured voice %r; using %s",
        configured_voice,
        DEFAULT_VOICE,
    )
    return DEFAULT_VOICE


def speech_parts(text, configured_voice=DEFAULT_VOICE):
    """Select the reply voice, switching sentences it cannot read; retain sentences for chunk retries."""
    configured_voice = validated_voice(configured_voice)
    language = detect_language(text)
    if language is None and _script_is_main(text, r"[\u0400-\u052f]+"):
        language = _language(configured_voice)
        if language not in ("ru", "uk"):
            language = "ru"
    voice = select_voice(language, configured_voice)
    logger.info("Speech language=%s voice=%s", language or "uncertain", voice)
    parts = []
    for segment in _segments(text):
        segment_voice = voice
        if not any(char.isalnum() for char in segment):
            if parts:
                segment_voice = parts[-1][1]
        else:
            segment_language = _switch_language(segment, voice)
            if segment_language:
                segment_voice = select_voice(segment_language, configured_voice)
                logger.info("Speech segment language=%s voice=%s", segment_language, segment_voice)
        parts.extend((piece, segment_voice) for piece in split_for_telegram(segment))
    return parts
