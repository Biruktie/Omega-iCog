"""Reported speech failures: real detection, mocked catalogue and transport."""
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import media_handler as mh
import speech_language as sl
from speech_text import prepare_speech
from text_splitter import split_for_telegram

VOICES = [
    {"Locale": locale, "ShortName": name, "Gender": gender}
    for locale, name, gender in [
        ("en-AU", "en-AU-WilliamMultilingualNeural", "Male"),
        ("en-US", sl.DEFAULT_VOICE, "Female"),
        ("en-US", "en-US-GuyNeural", "Male"),
        ("ru-RU", "ru-RU-DmitryNeural", "Male"),
        ("ru-RU", "ru-RU-SvetlanaNeural", "Female"),
        ("uk-UA", "uk-UA-PolinaNeural", "Female"),
        ("zh-CN", "zh-CN-XiaoxiaoNeural", "Female"),
        ("ja-JP", "ja-JP-NanamiNeural", "Female"),
        ("ar-AE", "ar-AE-FatimaNeural", "Female"),
        ("ar-SA", "ar-SA-ZariyahNeural", "Female"),
        ("ur-IN", "ur-IN-GulNeural", "Female"),
        ("ur-PK", "ur-PK-UzmaNeural", "Female"),
        ("he-IL", "he-IL-HilaNeural", "Female"),
        ("hi-IN", "hi-IN-SwaraNeural", "Female"),
        ("th-TH", "th-TH-PremwadeeNeural", "Female"),
        ("el-GR", "el-GR-AthinaNeural", "Female"),
        ("fr-BE", "fr-BE-CharlineNeural", "Female"),
        ("fr-FR", "fr-FR-DeniseNeural", "Female"),
        ("de-AT", "de-AT-IngridNeural", "Female"),
        ("de-DE", "de-DE-KatjaNeural", "Female"),
        ("es-AR", "es-AR-ElenaNeural", "Female"),
        ("es-ES", "es-ES-ElviraNeural", "Female"),
    ]
]


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = self.enterContext(patch.object(sl, "available_voices", return_value=VOICES))

    def test_reported_short_phrases(self):
        for text, language in [("Как дела?", "ru"), ("Да.", "ru"), ("你好。", "zh"),
                               ("Як справи?", "uk"), ("Доброе утро. Сегодня в столице ясно и холодно.", "ru")]:
            with self.subTest(text=text):
                self.assertTrue(all(voice.startswith(language + "-") for _, voice in sl.speech_parts(text)))

    def test_ambiguous_cyrillic_uses_context(self):
        self.assertTrue(all(voice.startswith("uk-") for _, voice in
                            sl.speech_parts("Да. Це український текст із літерами ї та є.")))

    def test_mixed_latin_does_not_override_cyrillic(self):
        text = "Я люблю Python и JavaScript."
        parts = sl.speech_parts(text)
        self.assertEqual("".join(piece for piece, _ in parts), text)
        self.assertTrue(all(voice.startswith("ru-") for _, voice in parts))
        chunks = list(sl.speech_chunks([(i, part, voice) for i, (part, voice) in enumerate(parts)]))
        self.assertEqual(len(chunks), 1)
        self.assertEqual("".join(part for _, part, _ in chunks[0]), text)

    def test_sentence_chunks_preserve_whole_sentences_and_share_voices(self):
        sentence = "This is an ordinary English sentence with several words. "
        text = sentence * 120
        parts = sl.speech_parts(text)
        chunks = list(sl.speech_chunks([(i, part, voice) for i, (part, voice) in enumerate(parts)]))
        texts = ["".join(part for _, part, _ in chunk) for chunk in chunks]
        self.assertEqual("".join(texts), text)
        self.assertGreater(len(texts), 1)
        self.assertLess(len(texts), 4)
        self.assertTrue(all(len(piece) <= 4096 and piece.endswith(". ") for piece in texts))
        mixed = [(0, "Hello. ", "en-US-AriaNeural"),
                 (1, "Я люблю Python и JavaScript. ", "ru-RU-SvetlanaNeural"),
                 (2, "Goodbye.", "en-US-AriaNeural")]
        self.assertEqual(list(sl.speech_chunks(mixed)), [mixed])

    def test_typo_uses_default_then_language_routing(self):
        self.assertEqual(sl.speech_parts("Hello there", "en-US-TypoNeural"),
                         [("Hello there", sl.DEFAULT_VOICE)])
        self.assertTrue(sl.speech_parts("Как дела?", "en-US-TypoNeural")[0][1].startswith("ru-"))

    def test_catalogue_outage_is_not_cached_as_invalid_voice(self):
        self.catalogue.side_effect = RuntimeError("offline")
        with self.assertRaises(RuntimeError):
            sl.validated_voice("en-US-GuyNeural")
        self.catalogue.side_effect = None
        self.assertEqual(sl.validated_voice("en-US-GuyNeural"), "en-US-GuyNeural")

    def test_long_line_preserves_words_and_text(self):
        text = "ordinary words " * 900
        parts = split_for_telegram(text)
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(len(piece) <= 4096 for piece in parts))
        self.assertTrue(all(piece[-1].isspace() for piece in parts[:-1]))
        rendered = split_for_telegram(text, lambda part: len(part) * 2 <= 4096)
        self.assertEqual("".join(rendered), text)
        self.assertTrue(all(len(piece) * 2 <= 4096 for piece in rendered))

    def test_cleanup_bare_domains_symbols_and_ordinary_text(self):
        self.assertEqual(prepare_speech("Read example.com/path ⭐ ⏰ ▶ ™ now"), "Read now")
        text = "Keep v2.0 3.14 report.pdf user@example.com 50% $20 2+2"
        self.assertEqual(prepare_speech(text), text)

    def test_non_latin_sentence_in_latin_reply_gets_a_voice_that_can_read_it(self):
        for text, voice in [
            ("In Russian, good morning is: Доброе утро.", "ru-RU-SvetlanaNeural"),
            ("Я пишу на Python, JavaScript, TypeScript, Go и Rust.", "ru-RU-SvetlanaNeural"),
            ("Tokyo is 東京 in Japanese.", "zh-CN-XiaoxiaoNeural"),
            ("The building is called 東京都庁舎です.", "ja-JP-NanamiNeural"),
            ("In Ukrainian, hello is: Привіт!", "uk-UA-PolinaNeural"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(sl.speech_parts(text), [(text, voice)])

    def test_single_letters_of_another_alphabet_do_not_switch_voice(self):
        for text in ("The value of π is about 3.14.", "The resistance is 5 Ω.",
                     "Alpha (α) and beta (β) particles."):
            with self.subTest(text=text):
                self.assertEqual(sl.speech_parts(text), [(text, sl.DEFAULT_VOICE)])
        text = "The character 水 means water."
        self.assertEqual(sl.speech_parts(text), [(text, "zh-CN-XiaoxiaoNeural")])

    def test_reply_voice_keeps_a_sentence_mostly_in_its_own_script(self):
        for text in ("Это русский ответ про греческий алфавит. Буква α называется альфа, а β называется бета.",
                     "Слово שלום означает мир, и его говорят при встрече."):
            with self.subTest(text=text):
                self.assertTrue(all(voice == "ru-RU-SvetlanaNeural" for _, voice in sl.speech_parts(text)))

    def test_closing_punctuation_stays_with_its_sentence(self):
        for text, voice in [("In Japanese, good morning is 「おはようございます。」", "ja-JP-NanamiNeural"),
                            ("In Chinese, hello is “你好。”", "zh-CN-XiaoxiaoNeural")]:
            with self.subTest(text=text):
                self.assertEqual(sl.speech_parts(text), [(text, voice)])

    def test_piece_without_letters_keeps_the_previous_voice(self):
        parts = sl.speech_parts("你好。\n+++\nHello there, how are you today?")
        self.assertEqual([voice for _, voice in parts],
                         ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoxiaoNeural", sl.DEFAULT_VOICE])

    def test_chinese_sentence_before_english_is_split_without_a_space(self):
        self.assertEqual(sl.speech_parts("你好。Hello, how are you today?"),
                         [("你好。", "zh-CN-XiaoxiaoNeural"), ("Hello, how are you today?", sl.DEFAULT_VOICE)])

    def test_equal_script_counts_pick_the_same_voice_every_time(self):
        self.assertEqual(sl.speech_parts("Say Да or 好吗 to them."),
                         [("Say Да or 好吗 to them.", "ru-RU-SvetlanaNeural")])

    def test_only_the_unreadable_sentence_changes_voice(self):
        text = "Good morning in Russian is said like this. Доброе утро. It means good morning."
        parts = sl.speech_parts(text)
        self.assertEqual("".join(piece for piece, _ in parts), text)
        self.assertEqual([voice for _, voice in parts],
                         [sl.DEFAULT_VOICE, "ru-RU-SvetlanaNeural", sl.DEFAULT_VOICE])

    def test_rerouted_sentence_keeps_the_configured_gender(self):
        self.assertEqual(sl.speech_parts("In Russian, good morning is: Доброе утро.", "en-US-GuyNeural"),
                         [("In Russian, good morning is: Доброе утро.", "ru-RU-DmitryNeural")])

    def test_one_word_in_other_scripts_gets_a_matching_voice(self):
        for text, voice in [("שלום", "he-IL-HilaNeural"), ("شكرا", "ar-SA-ZariyahNeural"),
                            ("नमस्ते", "hi-IN-SwaraNeural"), ("สวัสดี", "th-TH-PremwadeeNeural"),
                            ("Γεια", "el-GR-AthinaNeural")]:
            with self.subTest(text=text):
                self.assertEqual(sl.speech_parts(text), [(text, voice)])

    def test_language_switch_prefers_the_main_regional_voice(self):
        for language, voice in [("fr", "fr-FR-DeniseNeural"), ("de", "de-DE-KatjaNeural"),
                                ("es", "es-ES-ElviraNeural"), ("ar", "ar-SA-ZariyahNeural"),
                                ("ur", "ur-PK-UzmaNeural")]:
            with self.subTest(language=language):
                self.assertEqual(sl.select_voice(language, sl.DEFAULT_VOICE), voice)
        self.assertEqual(sl.select_voice("en", "ru-RU-DmitryNeural"), "en-US-GuyNeural")

    def test_chinese_splits_after_a_full_stop_without_spaces(self):
        text = "我们今天下午在图书馆里一起复习数学和物理。" * 300
        pieces = split_for_telegram(text)
        self.assertEqual("".join(pieces), text)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(piece.endswith("。") for piece in pieces))
        chunks = list(sl.speech_chunks([(i, piece, voice) for i, (piece, voice)
                                        in enumerate(sl.speech_parts(text))]))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("".join(piece for _, piece, _ in chunk).endswith("。") for chunk in chunks))


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(mh._voice_requests, {}, clear=True))
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {"MEMORY_DIR": self.directory}))
        config = types.ModuleType("config")
        config.config_get_by_key = lambda key, default=None: default
        self.enterContext(patch.dict(sys.modules, {"config": config}))
        self.channel = types.SimpleNamespace(chat_id=123, _reply_to_id=45, send_message=Mock())
        self.enterContext(patch.object(mh, "_live_channel", self.channel))
        self.enterContext(patch.object(mh, "_live_send_chat_action", None))
        self.enterContext(patch.object(mh, "_tts_allowed", return_value=True))
        self.enterContext(patch.object(mh, "_prompt_is_unsafe", return_value=False))
        self.enterContext(patch.object(mh, "speech_parts", return_value=[
            ("First.", sl.DEFAULT_VOICE), ("Second.", sl.DEFAULT_VOICE), ("Third.", sl.DEFAULT_VOICE)]))
        self.synth = self.enterContext(patch.object(mh, "_synthesise_speech", side_effect=[None, None, b"first", None, b"third"]))
        self.send = self.enterContext(patch.object(mh, "_live_send_voice", return_value=100))

    def test_partial_retry_only_sends_failed_part_and_new_request_is_distinct(self):
        self.assertTrue(mh.speak("request").startswith("VOICE_PARTIAL"))
        self.assertEqual(self.send.call_count, 2)
        self.assertEqual([c.args[0] for c in self.send.call_args_list], [b"first", b"third"])
        self.channel.send_message.assert_called_once()
        mh.next_turn()
        self.synth.side_effect = [b"second"]
        self.assertEqual(mh.speak("request"), "VOICE_SENT")
        self.assertEqual(self.send.call_count, 3)
        self.assertEqual(self.send.call_args.args[0], b"second")
        mh.next_turn()
        self.assertIn("already delivered", mh.speak("request"))
        self.assertEqual(self.send.call_count, 3)
        self.channel._reply_to_id = 46
        self.synth.side_effect = None
        self.synth.return_value = b"new"
        self.assertEqual(mh.speak("request"), "VOICE_SENT")
        self.assertEqual(self.send.call_count, 4)

    def test_uncertain_upload_is_not_retried(self):
        self.synth.side_effect = None
        self.synth.return_value = b"audio"
        self.send.side_effect = TimeoutError()
        self.assertTrue(mh.speak("request").startswith("VOICE_FAILED"))
        mh.next_turn()
        self.assertIn("delivery uncertain", mh.speak("request"))
        self.assertEqual(self.send.call_count, 1)
        self.channel.send_message.assert_called_once()

    def test_text_budget_splits_uploads_without_repeating_them(self):
        self.synth.side_effect = [b"aaaa", b"bbbb"]
        with patch.object(mh, "speech_parts", return_value=[
                ("a" * 2000 + ". ", sl.DEFAULT_VOICE),
                ("b" * 2000 + ". ", sl.DEFAULT_VOICE),
                ("c" * 2000 + ".", sl.DEFAULT_VOICE)]):
            self.assertEqual(mh.speak("request"), "VOICE_SENT")
            self.assertEqual([call.args[0] for call in self.send.call_args_list],
                             [b"aaaa", b"bbbb"])
            mh.next_turn()
            self.assertIn("already delivered", mh.speak("request"))
            self.assertEqual(self.send.call_count, 2)

    def test_successful_chunk_is_synthesized_once(self):
        self.synth.side_effect = None
        self.synth.return_value = b"audio"
        self.assertEqual(mh.speak("request"), "VOICE_SENT")
        self.synth.assert_called_once_with("First.Second.Third.", sl.DEFAULT_VOICE)
        self.send.assert_called_once_with(b"audio")

    def test_voice_changes_inside_a_chunk_make_one_voice_message(self):
        self.synth.side_effect = [b"A1", b"R", b"A2"]
        with patch.object(mh, "speech_parts", return_value=[
                ("Hello. ", sl.DEFAULT_VOICE), ("Доброе утро. ", "ru-RU-SvetlanaNeural"),
                ("Bye.", sl.DEFAULT_VOICE)]):
            self.assertEqual(mh.speak("request"), "VOICE_SENT")
        self.assertEqual([c.args for c in self.synth.call_args_list],
                         [("Hello. ", sl.DEFAULT_VOICE), ("Доброе утро. ", "ru-RU-SvetlanaNeural"),
                          ("Bye.", sl.DEFAULT_VOICE)])
        self.send.assert_called_once_with(b"A1RA2")

    def test_same_text_twice_in_one_turn_is_sent_twice(self):
        self.synth.side_effect = None
        self.synth.return_value = b"audio"
        self.assertEqual(mh.speak("Good morning"), "VOICE_SENT")
        self.assertEqual(mh.speak("Good morning"), "VOICE_SENT")
        self.assertEqual(self.send.call_count, 2)
        mh.next_turn()
        self.assertIn("already delivered 2 times", mh.speak("Good morning"))
        self.assertEqual(self.send.call_count, 2)

    def test_same_turn_repeat_after_a_partial_attempt_is_a_full_repeat(self):
        self.synth.side_effect = [None, None, b"a", None, b"ab"]
        with patch.object(mh, "speech_parts", return_value=[("A. ", sl.DEFAULT_VOICE), ("B.", sl.DEFAULT_VOICE)]):
            self.assertTrue(mh.speak("request").startswith("VOICE_PARTIAL"))
            self.assertEqual(mh.speak("request"), "VOICE_SENT")
        self.assertEqual(self.synth.call_args_list[-1].args, ("A. B.", sl.DEFAULT_VOICE))
        self.assertEqual([c.args[0] for c in self.send.call_args_list], [b"a", b"ab"])

    def test_repeat_in_a_later_turn_without_a_new_message_is_not_sent(self):
        self.synth.side_effect = None
        self.synth.return_value = b"audio"
        self.assertEqual(mh.speak("Good morning"), "VOICE_SENT")
        mh.next_turn()
        status = mh.speak("Good morning")
        self.assertTrue(status.startswith("VOICE_DUPLICATE"), status)
        self.assertIn("already delivered once", status)
        self.assertIn("not sent again", status)
        self.assertEqual(self.send.call_count, 1)

    def test_one_failed_synthesis_is_retried_as_a_whole(self):
        self.synth.side_effect = [None, b"audio"]
        self.assertEqual(mh.speak("request"), "VOICE_SENT")
        self.send.assert_called_once_with(b"audio")
        self.channel.send_message.assert_not_called()

    def test_persistent_failure_sends_surviving_sentences_together(self):
        self.synth.side_effect = [None, None, b"a", b"b", None, b"d"]
        with patch.object(mh, "speech_parts", return_value=[
                ("A. ", sl.DEFAULT_VOICE), ("B. ", sl.DEFAULT_VOICE),
                ("C. ", sl.DEFAULT_VOICE), ("D.", sl.DEFAULT_VOICE)]):
            self.assertTrue(mh.speak("request").startswith("VOICE_PARTIAL"))
        self.assertEqual([c.args[0] for c in self.send.call_args_list], [b"ab", b"d"])

    def test_checkpoints_are_shared_across_instances_in_this_process(self):
        from media_handler import SpeechDelivery
        first = SpeechDelivery("request-id")
        first.set(0, "sent", 100)
        first.set(1, "uploading")
        second = SpeechDelivery("request-id")
        self.assertEqual(second.get(0), "sent")
        self.assertEqual(second.get(1), "uploading")

    def test_no_database_or_other_file_is_created(self):
        mh.SpeechDelivery("request-id").set_many([0, 1], "sent", 123)
        self.assertEqual(list(Path(self.directory).iterdir()), [])

    def test_cache_is_bounded_and_evicts_least_recently_used_request(self):
        with patch.object(mh, "MAX_VOICE_REQUESTS", 2):
            first = mh.SpeechDelivery("first")
            first.set(0, "sent", 100)
            mh.SpeechDelivery("second").set(0, "uncertain")
            self.assertEqual(first.get(0), "sent")  # refresh first
            mh.SpeechDelivery("third").set(0, "failed")
            self.assertEqual(list(mh._voice_requests), ["first", "third"])
            self.assertEqual(mh.SpeechDelivery("second").get(0), "pending")
            self.assertEqual(len(mh._voice_requests), 2)

    def test_cleared_cache_loses_retry_state(self):
        mh.SpeechDelivery("request-id").set_many([0, 1], "sent", 123)
        mh._voice_requests.clear()  # equivalent to starting with a fresh process
        self.assertEqual(mh.SpeechDelivery("request-id").get(0), "pending")

    def test_group_updates_keep_message_id_and_other_requests_isolated(self):
        first = mh.SpeechDelivery("first")
        first.set_many([0, 1], "sent", 123)
        self.assertEqual(first.get(0), "sent")
        self.assertEqual(first.get(1), "sent")
        self.assertEqual(mh._voice_requests["first"][1]["message_id"], 123)
        self.assertEqual(mh.SpeechDelivery("second").get(0), "pending")


if __name__ == "__main__":
    unittest.main()
