"""Real offline detection; mock the network catalogue and audio delivery."""
import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speech_language as sl
import media_handler as mh

VOICES = [
    {"Locale": "en-US", "ShortName": sl.DEFAULT_VOICE, "Gender": "Female"},
    {"Locale": "en-US", "ShortName": "en-US-GuyNeural", "Gender": "Male"},
    {"Locale": "en-GB", "ShortName": "en-GB-SoniaNeural", "Gender": "Female"},
    {"Locale": "nb-NO", "ShortName": "nb-NO-PernilleNeural", "Gender": "Female"},
    {"Locale": "ru-RU", "ShortName": "ru-RU-DmitryNeural", "Gender": "Male"},
    {"Locale": "ru-RU", "ShortName": "ru-RU-SvetlanaNeural", "Gender": "Female"},
    {"Locale": "uk-UA", "ShortName": "uk-UA-PolinaNeural", "Gender": "Female"},
    {"Locale": "fr-FR", "ShortName": "fr-FR-DeniseNeural", "Gender": "Female"},
    {"Locale": "fr-FR", "ShortName": "fr-FR-AlainNeural", "Gender": "Male"},
]
RUSSIAN = "Это проверка русского голоса. Пожалуйста, прочитайте весь текст вслух."


class SpeechLanguageTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(mh, "speech_chunks", side_effect=lambda parts: [[p] for p in parts]))
        self.catalogue = self.enterContext(patch.object(
            sl, "available_voices", return_value=VOICES))

    def test_real_language_detection(self):
        for text, expected in [
            ("Hello, this is an English sentence.", "en"),
            ("Привет, мир!", "ru"),
            (RUSSIAN, "ru"),
            ("Це перевірка української мови. Будь ласка, прочитайте цей текст уголос.", "uk"),
            ("Bonjour, comment allez-vous aujourd’hui ?", "fr"),
            ("Hola, me gustaría escuchar este texto en español.", "es"),
            ("こんにちは、今日はいい天気ですね。", "ja"),
            ("OK", None), ("12345", None), ("", None),
        ]:
            with self.subTest(text=text):
                self.assertEqual(sl.detect_language(text), expected)

    def test_matching_configured_voice_and_uncertainty_need_no_catalogue(self):
        for language, voice in [("en", "en-GB-SoniaNeural"),
                                ("ru", "ru-RU-SvetlanaNeural"),
                                (None, "en-US-AriaNeural"),
                                ("no", "nb-NO-PernilleNeural")]:
            self.assertEqual(sl.select_voice(language, voice), voice)
        self.catalogue.assert_not_called()

    def test_russian_selection_and_return_to_english(self):
        self.assertEqual(sl.select_voice("ru", sl.DEFAULT_VOICE), "ru-RU-SvetlanaNeural")
        self.assertEqual(sl.select_voice("en", "ru-RU-DmitryNeural"), "en-US-GuyNeural")

    def test_unsupported_language_falls_back_with_warning(self):
        with self.assertLogs(sl.logger, level="WARNING") as logs:
            self.assertEqual(sl.select_voice("la", sl.DEFAULT_VOICE), sl.DEFAULT_VOICE)
        self.assertIn("detected language la", logs.output[0])

    def test_uncertain_short_text_uses_configured_voice(self):
        self.assertEqual(sl.speech_parts("OK", "en-GB-SoniaNeural"),
                         [("OK", "en-GB-SoniaNeural")])
        self.catalogue.assert_called()

    def test_alphabetically_first_male_voice_is_skipped(self):
        self.assertEqual(sl.select_voice("fr", sl.DEFAULT_VOICE), "fr-FR-DeniseNeural")

    def test_configured_male_voice_is_preserved_for_matching_or_uncertain_text(self):
        for language in ("ru", None):
            self.assertEqual(sl.select_voice(language, "ru-RU-DmitryNeural"),
                             "ru-RU-DmitryNeural")
        self.catalogue.assert_not_called()

    def test_missing_or_male_gender_is_not_a_fallback(self):
        for gender in ("Male", None):
            self.catalogue.return_value = [
                VOICES[0],
                {"Locale": "ru-RU", "ShortName": "ru-RU-SvetlanaNeural", "Gender": gender}]
            self.assertEqual(sl.select_voice("ru", sl.DEFAULT_VOICE), sl.DEFAULT_VOICE)

    def test_whole_reply_keeps_the_male_gender(self):
        text = "Hello, this is an English sentence.\n" + RUSSIAN + "\nHello, this is an English sentence."
        parts = sl.speech_parts(text, "en-US-GuyNeural")
        self.assertEqual([voice for _, voice in parts],
                         ["en-US-GuyNeural", "ru-RU-DmitryNeural", "ru-RU-DmitryNeural", "en-US-GuyNeural"])

    def test_male_configuration_does_not_fall_back_to_female(self):
        self.assertEqual(sl.select_voice("uk", "en-US-GuyNeural"), "en-US-GuyNeural")

    def test_misdetected_line_does_not_cancel_speech(self):
        text = "Hello, this is an English sentence.\nOmega Claw v2\nПривет, мир!\nThis is the final English sentence."
        config = types.ModuleType("config")
        config.config_get_by_key = lambda key, default=None: default
        # Only the whole reply is classified, never its individual lines.
        with patch.object(sl, "detect_language", return_value="en") as detect, \
                patch.dict(sys.modules, {"config": config}), \
                patch.object(mh, "_tts_allowed", return_value=True), \
                patch.object(mh, "_prompt_is_unsafe", return_value=False), \
                patch.object(mh, "_live_send_chat_action", None), \
                patch.object(mh, "_synthesise_speech", return_value=b"audio") as synth, \
                patch.object(mh, "_live_send_voice") as send:
            self.assertEqual(mh.speak(text), "VOICE_SENT")
            self.assertEqual([call.args[1] for call in synth.call_args_list],
                             [sl.DEFAULT_VOICE, sl.DEFAULT_VOICE, "ru-RU-SvetlanaNeural", sl.DEFAULT_VOICE])
            detect.assert_called_once_with(text)
            self.assertEqual("\n".join(call.args[0].strip() for call in synth.call_args_list), text)
            self.assertEqual(send.call_count, 4)

    def test_unknown_configured_gender_fails_on_language_switch(self):
        for configured in ("en-US-UnknownNeural", sl.DEFAULT_VOICE):
            with self.subTest(configured=configured):
                self.catalogue.return_value = [
                    {"Locale": "en-US", "ShortName": sl.DEFAULT_VOICE},
                    VOICES[4],
                ]
                with self.assertRaisesRegex(ValueError, "Cannot determine gender"):
                    sl.select_voice("ru", configured)

    def test_mixed_lines_keep_order(self):
        text = "Hello, this is an English sentence.\n" + RUSSIAN + "\nHello, this is an English sentence."
        parts = sl.speech_parts(text)
        self.assertEqual([voice for _, voice in parts],
                         [sl.DEFAULT_VOICE, "ru-RU-SvetlanaNeural", "ru-RU-SvetlanaNeural", sl.DEFAULT_VOICE])

    def test_main_language_is_not_the_first_phrase_or_first_4096_characters(self):
        english = "The weather is sunny today and we are going to walk through the park together. "
        for prefix, first_voice in (("Привет. ", "ru-RU-DmitryNeural"), ("你好。 ", "en-US-GuyNeural"),
                                    ("Bonjour. ", "en-US-GuyNeural")):
            text = prefix + english * 100
            with self.subTest(prefix=prefix):
                self.assertEqual(sl.detect_language(text), "en")
                parts = sl.speech_parts(text, "en-US-GuyNeural")
                self.assertEqual(parts[0], (prefix, first_voice))
                self.assertTrue(all(voice == "en-US-GuyNeural" for _, voice in parts[1:]))

    def test_one_detection_and_voice_for_all_sentences(self):
        text = "Hello. Привет. Bonjour."
        with patch.object(sl, "detect_language", return_value="ru") as detect:
            parts = sl.speech_parts(text)
        detect.assert_called_once_with(text)
        self.assertEqual({voice for _, voice in parts}, {"ru-RU-SvetlanaNeural"})
        self.assertEqual("".join(piece for piece, _ in parts), text)
        self.assertEqual(" ".join(piece.strip() for piece, _ in parts),
                         text.replace("\n", " "))

    def test_long_russian_text_still_splits(self):
        text = (RUSSIAN + " ") * 100
        parts = sl.speech_parts(text)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(0 < len(piece) <= 4096 for piece, _ in parts))
        self.assertTrue(all(voice == "ru-RU-SvetlanaNeural" for _, voice in parts))
        self.assertEqual("".join(piece for piece, _ in parts), text)

    def test_speak_routes_cleaned_text_to_synthesis(self):
        config = types.ModuleType("config")
        config.config_get_by_key = lambda key, default=None: default
        with patch.dict(sys.modules, {"config": config}), \
                patch.object(mh, "_tts_allowed", return_value=True), \
                patch.object(mh, "_prompt_is_unsafe", return_value=False), \
                patch.object(mh, "_live_send_chat_action", None), \
                patch.object(mh, "_synthesise_speech", return_value=b"audio") as synth, \
                patch.object(mh, "_live_send_voice") as send:
            self.assertEqual(mh.speak("**Привет, мир!** https://example.com"), "VOICE_SENT")
            synth.assert_called_once_with("Привет, мир!", "ru-RU-SvetlanaNeural")
            send.assert_called_once()
            synth.reset_mock()
            send.reset_mock()
            self.catalogue.side_effect = RuntimeError("catalogue unavailable")
            self.assertIn("VOICE_FAILED", mh.speak(RUSSIAN))
            synth.assert_not_called()
            send.assert_not_called()


class VoiceCatalogueTests(unittest.TestCase):
    def setUp(self):
        sl._clear_available_voices_cache()
        self.addCleanup(sl._clear_available_voices_cache)

    def test_success_is_cached(self):
        with patch("edge_tts.list_voices", new_callable=AsyncMock,
                   return_value=VOICES) as fetch:
            self.assertEqual(sl.available_voices(), VOICES)
            self.assertEqual(sl.available_voices(), VOICES)
            fetch.assert_awaited_once()

    def test_unavailable_catalogue_uses_cooldown_then_retries(self):
        for outcome in (TimeoutError("timeout"), []):
            with self.subTest(outcome=outcome):
                sl._clear_available_voices_cache()
                with patch.object(sl, "monotonic", side_effect=[100, 100, 100, 131]), \
                        patch("edge_tts.list_voices", new_callable=AsyncMock,
                              side_effect=[outcome, VOICES]) as fetch:
                    self.assertEqual(sl.available_voices(), [])
                    self.assertEqual(sl.available_voices(), [])
                    fetch.assert_awaited_once()

                    self.assertEqual(sl.available_voices(), VOICES)
                    self.assertEqual(sl.available_voices(), VOICES)
                    self.assertEqual(fetch.await_count, 2)

    def test_clear_cache_resets_retry_cooldown(self):
        with patch.object(sl, "monotonic", return_value=100), \
                patch("edge_tts.list_voices", new_callable=AsyncMock,
                    side_effect=[TimeoutError("timeout"), VOICES]) as fetch:

            self.assertEqual(sl.available_voices(), [])
            fetch.assert_awaited_once()

            sl._clear_available_voices_cache()

            self.assertEqual(sl.available_voices(), VOICES)
            self.assertEqual(fetch.await_count, 2)

    def test_lookup_inside_running_event_loop(self):
        async def lookup():
            return sl.available_voices()

        with patch("edge_tts.list_voices", new_callable=AsyncMock,
                   return_value=VOICES):
            self.assertEqual(asyncio.run(lookup()), VOICES)


if __name__ == "__main__":
    unittest.main()
