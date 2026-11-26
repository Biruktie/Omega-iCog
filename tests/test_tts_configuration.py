"""Voice configuration and launcher regression tests, runnable with unittest."""
import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class VoiceConfigurationTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("voice_config", ROOT / "src/config.py")
        self.config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.config)
        self.config._CONFIG_FILE = {"EDGE_TTS_VOICE": "yaml-voice"}
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_voice_precedence(self):
        for command, prefixed, bare, expected in (
            ("argument", "prefixed", "bare", "argument"),
            (None, "prefixed", "bare", "prefixed"),
            (None, "", "bare", "bare"),
            (None, "", "", "yaml-voice"),
        ):
            with self.subTest(expected=expected):
                self.config._CONFIG = {}
                self.config._COMMAND_LINE = {} if command is None else {"EDGE_TTS_VOICE": command}
                os.environ.update(OMEGA_EDGE_TTS_VOICE=prefixed, EDGE_TTS_VOICE=bare)
                self.assertEqual(self.config.config_get_by_key("EDGE_TTS_VOICE", "default"), expected)

    def test_unrelated_keys_still_require_prefix(self):
        os.environ["provider"] = "bare-ignored"
        self.assertEqual(self.config.config_get_by_key("provider", "default"), "default")

    def test_launcher_forwards_voice(self):
        for prefixed, expected in (("", "ru-RU-SvetlanaNeural"), ("en-US-GuyNeural", "en-US-GuyNeural")):
            with self.subTest(prefixed=prefixed):
                env = dict(os.environ, PATH=os.defpath, TEST_SERVER_IP="127.0.0.1",
                           EDGE_TTS_VOICE="ru-RU-SvetlanaNeural", OMEGA_EDGE_TTS_VOICE=prefixed)
                result = subprocess.run(
                    ["bash", "-c", 'docker() { printf "%s\\n" "$@"; }; source scripts/omega start -p Test -t test'],
                    cwd=ROOT, env=env, text=True, capture_output=True, check=True)
                self.assertIn("OMEGA_EDGE_TTS_VOICE=" + expected, result.stdout.splitlines())

    def test_entrypoint_scrubbing_preserves_only_allowed_settings(self):
        script = (ROOT / "entrypoint.sh").read_text()
        # Execute the actual environment-filtering block without starting services.
        block = script[script.index('SAFE_VARS='):script.index('exec env -i')]
        result = subprocess.run(
            ["bash", "-c", block + '\n/usr/bin/env -i $env_args /usr/bin/env'],
            env={"EDGE_TTS_VOICE": "ru-RU-SvetlanaNeural", "OMEGA_EDGE_TTS_VOICE": "en-US-GuyNeural",
                 "OPENAI_API_KEY": "test-secret"}, text=True, capture_output=True, check=True)
        self.assertIn("EDGE_TTS_VOICE=ru-RU-SvetlanaNeural", result.stdout.splitlines())
        self.assertIn("OMEGA_EDGE_TTS_VOICE=en-US-GuyNeural", result.stdout.splitlines())
        self.assertNotIn("test-secret", result.stdout)

    def test_fallback_provider_defaults_when_environment_is_empty(self):
        self.config._CONFIG = {}
        self.config._COMMAND_LINE = {}
        self.config._CONFIG_FILE = {}

        os.environ["TTS_FALLBACK_PROVIDER"] = ""

        self.assertEqual(
            self.config.config_get_by_key("TTS_FALLBACK_PROVIDER", "google"),
            "google",
        )


    def test_fallback_provider_accepts_explicit_provider(self):
        for variable, value in (
            ("OMEGA_TTS_FALLBACK_PROVIDER", "edge"),
            ("TTS_FALLBACK_PROVIDER", "google"),
        ):
            with self.subTest(variable=variable):
                self.config._CONFIG = {}
                os.environ.clear()
                os.environ[variable] = value

                self.assertEqual(
                    self.config.config_get_by_key("TTS_FALLBACK_PROVIDER", "google"),
                    value,
                )


    def test_fallback_provider_can_be_explicitly_disabled(self):
        for value in ("none", "off", "disabled"):
            with self.subTest(value=value):
                self.config._CONFIG = {}
                os.environ.clear()
                os.environ["TTS_FALLBACK_PROVIDER"] = value

                self.assertEqual(
                    self.config.config_get_by_key("TTS_FALLBACK_PROVIDER", "google"),
                    "",
                )


    def test_empty_fallback_environment_uses_config_file(self):
        self.config._CONFIG = {}
        self.config._COMMAND_LINE = {}
        self.config._CONFIG_FILE = {"TTS_FALLBACK_PROVIDER": "edge"}

        os.environ["TTS_FALLBACK_PROVIDER"] = ""

        self.assertEqual(
            self.config.config_get_by_key("TTS_FALLBACK_PROVIDER", "google"),
            "edge",
        )

    def test_fallback_provider_command_line_disable(self):
        self.config._CONFIG = {}
        self.config._CONFIG_FILE = {}
        self.config._COMMAND_LINE = {
            "TTS_FALLBACK_PROVIDER": "off"
        }

        self.assertEqual(
            self.config.config_get_by_key("TTS_FALLBACK_PROVIDER", "google"),
            "",
        )

    def test_launcher_forwards_disabled_fallback_provider(self):
        env = dict(
            os.environ,
            PATH=os.defpath,
            TEST_SERVER_IP="127.0.0.1",
            OMEGA_TTS_FALLBACK_PROVIDER="none",
        )

        result = subprocess.run(
            [
                "bash", "-c",
                'docker() { printf "%s\\n" "$@"; }; source scripts/omega start -p Test -t test'
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn(
            "OMEGA_TTS_FALLBACK_PROVIDER=none",
            result.stdout.splitlines(),
        )


if __name__ == "__main__":
    unittest.main()
