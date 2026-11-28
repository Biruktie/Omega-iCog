# omega-telegram

A Telegram communication channel for Omega with media support: the agent can
read images, PDFs and voice notes that users attach, generate images, and send
voice replies when enabled.

This is the `telegram` channel. It replaces core's earlier HTTP-polling Telegram
channel and keeps everything that one provided — the gateway proxy path, the
outbound retry queue, and the channel auth handshake.

## Capabilities

| Capability | How the agent sees it |
| --- | --- |
| Inbound photo | Message shows `[image]`; the agent calls `describe-image` on demand |
| Inbound PDF | Extracted text is inlined into the message |
| Inbound voice / audio | Whisper transcript is inlined into the message |
| Outbound image | The agent calls `generate-image`, which generates and sends the photo |
| Outbound voice | The agent calls `speak`, which sends speech in ordered voice messages when enabled; long text uses the same paragraph/line splitting as text replies |
| Admin commands | `/kill`, `/pause [chat_id]`, `/togglesearch`, `/purge` (admin IDs only) |
| Safety | Ethics classification on inbound and outbound text, per-user spam throttling |
| Scope limits | `prompt.txt` is added to the agent's prompt as its own section |
| Authorization | `telegram_profile.yaml` chat/DM rules, plus core's auth handshake when enabled |
| Delivery | Outbound messages queue until the bot connects, and are retried on failure |

## Install

**1. Register the plugin** in core's `config/plugins.yaml`:

```yaml
- name: telegram
  loader: metta
  location: "{REPO}/plugins/telegram"
```

The `metta` loader is what lets `telegram.metta` register the plugin's skills and
its prompt section at load time. Core needs no telegram-specific code: the two
media skills are added with `add-skill`, not written into `src/skills.metta`.

**2. Dependencies** are in `requirements.txt`; core's Dockerfile installs every
`plugins/*/requirements.txt` automatically.

**3. Point at your own config if you want to** — `TG_PROFILE_PATH` and
`TG_POLICY_PATH` override where the plugin reads `telegram_profile.yaml` and
`policy.md`, resolved by core from the command line, an `OMEGA_`-prefixed
environment variable, or `config.yaml`. The files shipped here are the defaults;
mounting over them works too and needs no configuration.

**4. Fill in `telegram_profile.yaml`** — it ships permissive defaults that are
**not safe for production**. Set `admin_controls.admin_ids` to the Telegram user
IDs allowed to run admin commands, and `telegram.allowed_chats` to the chat IDs
the bot may operate in. Both are empty by default. Voice replies are opt-in;
set `telegram.reply_constraints.allow_voice_reply: true` to enable them.

## Use

```sh
sh run.sh run.metta commchannel=telegram provider=OpenRouter \
    embeddingprovider=Local TG_CHAT_ID="$TG_CHAT_ID"
```

`TG_CHAT_ID` is a run argument; `TG_BOT_TOKEN` is read from the environment. Run
only one instance per bot token — a second makes Telegram reject both.

When `GATEWAY_URL` is set every outbound call goes to that proxy, which holds the
credentials, and none of the keys below need to be in the agent's environment —
the container entrypoint scrubs them precisely so they are not. Telegram uses
two routes, `/telegram/` for API methods and `/telegram-file/` for downloads,
because Telegram serves files from a different path prefix. Vision, image
generation, transcription and moderation use `/anthropic/`, `/openrouter/` and
`/openai/`.

The keys in the table below are what a **direct** run needs. Behind the proxy
they belong to the proxy, not here.

| Variable | Required | Purpose |
| --- | --- | --- |
| `TG_BOT_TOKEN` | yes | Telegram bot token |
| `ANTHROPIC_API_KEY` | for vision | Used by the default vision provider |
| `OPENROUTER_API_KEY` | for image gen + Whisper | Also the vision key if `VISION_PROVIDER=OpenRouter` |
| `OPENAI_API_KEY` | for safety checks | Moderation API; without it the ethics passes allow content through |
| `EDGE_TTS_VOICE` | no | Preferred/fallback speech voice; also accepts `OMEGA_EDGE_TTS_VOICE`. Defaults to `en-US-AriaNeural`. Other detected languages automatically use a matching voice. |
| `VISION_PROVIDER` | no | `Anthropic` (default) or `OpenRouter` |
| `VISION_MODEL` | no | Overrides the provider's default vision model |
| `IMAGE_PROVIDER` | no | `OpenRouter` (default, FLUX) or `OpenAI` |
| `IMAGE_MODEL` | no | Overrides the image provider's default model |
| `TG_PROFILE_PATH` | no | Path to the channel profile, defaults to the shipped one |
| `TG_POLICY_PATH` | no | Path to the user-facing policy text, defaults to the shipped one |
| `TG_PROMPT_PATH` | no | Path to the prompt section, defaults to the shipped one |

## Voice replies

Voice precedence is runtime `EDGE_TTS_VOICE=...`, then environment
`OMEGA_EDGE_TTS_VOICE`, then environment `EDGE_TTS_VOICE`, then the
`EDGE_TTS_VOICE` YAML setting. Empty environment values are ignored.
Both environment names survive the container entrypoint. `scripts/omega`
forwards the voice from its environment; staging/production deployments read
the GitHub Actions variable `EDGE_TTS_VOICE`. Restart after changing the voice.

Speech synthesis uses **Edge TTS first**, with automatic fallback to **Google TTS** if Edge synthesis fails, times out, or returns empty audio. If the voice catalogue is unavailable, the configured Edge voice is retained so synthesis can still proceed.

No provider-selection configuration is required. The fallback happens automatically.

The `EDGE_TTS_VOICE` setting can still be used to configure the preferred Edge TTS voice.

- The whole reply's main language selects the reply voice, preserving configured
  gender. A sentence written mostly in a script that voice cannot read, such as
  Russian or Chinese inside an English reply, gets a voice of its own language in
  the same voice message. Single letters of another alphabet (α, π) keep the reply
  voice. Unsupported language/gender combinations use the configured voice; invalid
  voice names fall back to `en-US-AriaNeural` with a warning.
- Markdown, image descriptions, links and emoji are removed from speech, not text
  replies. Link labels, paragraph breaks, bare domain names such as `example.com`
  and ordered list numbers remain. Empty cleaned input returns `VOICE_INVALID_INPUT`.
- Sentences are grouped into text chunks of at most 4096 characters. Chunks end at
  sentence boundaries, and `。！？` need no following space; oversized sentences
  split at whitespace, with hard cuts only for oversized tokens. A failed chunk is
  retried once, then sentence by sentence, sending consecutive successful sentences
  together. The recording indicator refreshes throughout processing.
- Failed segments produce a text notice without discarding successful audio.
  The same text spoken twice within one agent turn is sent twice. In a later turn,
  an exact-text retry for the same inbound message sends only what was not
  delivered, and a fully delivered request returns `VOICE_DUPLICATE` with the
  number of copies delivered, without sending. Uncertain uploads
  are never automatically repeated while tracked. A voice-only
  in-memory cache retains the 100 most recently used requests; restart or eviction
  loses retry protection. No database is used. New messages and paraphrased
  requests are not deduplicated; existing SQLite files are left untouched.

## Location

The plugin does not have to live inside the core tree. It finds its own files
relative to its module, and reaches core's Python only through modules core puts
on the path, so `location` in `plugins.yaml` can be any absolute path:

```yaml
- name: telegram
  loader: metta
  location: "/opt/omega-telegram"
```

Verified by loading it from outside the repository with the in-tree copy removed.

## Tests

```sh
for f in tests/test_*.py; do python3 "$f"; done
```

Plain asserts, no framework. Network and Telegram calls are stubbed, so no
credentials are needed.
