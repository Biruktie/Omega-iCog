"""Plain spoken text for TTS; leaves the original chat text untouched."""
from html.parser import HTMLParser
import re

import pyromark
import emoji

_URL = re.compile(r"""
    (?<![\w@./-])
    (?:
        (?:https?://|www\.)[^\s<>"'\])}]+
        |
        (?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+
        (?!(?:js|ts|py|rb|php|java|kt|cs|cpp|md|txt|pdf|docx?|xlsx?|csv|json|ya?ml|toml|ini|cfg|log|sh|html?|css|xml|png|jpe?g|gif|svg|mp[34]|zip)\b)
        [a-z]{2,}
        (?:
            (?::[0-9]+)?(?:/|[?\#](?=[^\s.,;:!?]))[^\s<>"'\])}]*
            |
            :[0-9]+(?![\w.])
        )
    )
""", re.IGNORECASE | re.VERBOSE)


def _remove_urls(text):
    # Retain punctuation after a URL, e.g. "Read example.com/docs." -> "Read .".
    return _URL.sub(lambda match: match[0][len(match[0].rstrip(".,;:!?")):], text)


class _HTMLText(HTMLParser):
    """Extract data from raw HTML events without treating Markdown text as HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "div", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "div", "li"}:
            self.parts.append("\n")


def _markdown_text(text):
    """Keep textual parser events, omitting image subtrees and destinations."""
    source = text.encode("utf-8")  # Parser ranges are UTF-8 byte offsets.
    parts = []
    previous_end = 0
    image_depth = 0
    lists = []
    options = (pyromark.Options.ENABLE_STRIKETHROUGH
               | pyromark.Options.ENABLE_TABLES
               | pyromark.Options.ENABLE_TASKLISTS)
    for event, span in pyromark.events_with_range(text, options=options):
        end = span["end"]
        match event:
            case {"Start": {"Image": _}}:
                image_depth += 1
                continue
            case {"End": "Image"}:
                image_depth -= 1
                continue
            case _ if image_depth:
                continue
            case {"Start": {"List": first}}:
                lists.append(first is not None)
                continue
            case {"End": {"List": _}}:
                lists.pop()
                continue
            case {"Start": "Item"} if lists and lists[-1]:
                marker = re.match(rb"[\s>]*(\d{1,9}[.)])", source[span["start"]:])
                if not marker:
                    continue
                content = marker.group(1).decode() + " "
                end = span["start"]
            case {"Text": content} | {"Code": content}:
                pass
            case {"Html": content} | {"InlineHtml": content}:
                parser = _HTMLText()
                parser.feed(content)
                parser.close()
                content = "".join(parser.parts)
            case _:
                continue
        # Source gaps retain line/paragraph breaks without interpreting their
        # Markdown delimiters. Spaces separate cells in a Markdown table.
        if parts:
            gap = source[previous_end:span["start"]]
            if b"\n" in gap:
                parts.append("\n" * gap.count(b"\n"))
            elif b"|" in gap:
                parts.append(" ")
        parts.append(content)
        previous_end = end
    return "".join(parts)


def prepare_speech(text):
    text = _markdown_text(text.replace("\\n", "\n"))
    # Bare URLs are speech policy, not Markdown syntax.
    text = _remove_urls(text)
    # Emoji, their joiners/modifiers and keycap sequences, preserving ordinary
    # digits, punctuation and non-Latin letters.
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[\U0001f000-\U0001faff\u2600-\u27bf\u200d\ufe0f\u20e3]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines()).strip()
    return text if any(char.isalnum() for char in text) else ""
