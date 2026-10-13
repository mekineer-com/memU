from __future__ import annotations

import re
from collections.abc import Collection


_START = r"(?:[A-Za-z_:]|[^\x00-\x7f])"
_NAME = _START + r"(?:[A-Za-z0-9_.:-]|[^\x00-\x7f])*"
# Shield opaque XML and tag candidates; the caller's parser validates grammar.
XML_TOKEN = re.compile(
    r"(?P<opaque><!\[CDATA\[.*?\]\]>|<!--.*?-->|<\?.*?\?>)"
    rf"""|(?P<tag></?{_START}(?:[^<>"']|"[^"]*"|'[^']*')*>)"""
    rf"|(?P<damaged></?{_NAME}(?=\s+{_NAME}\s*=|[/=>])[^<>]*)"
    rf"|(?P<bare>&(?!{_NAME};|#[^;\s<&]*;)|<)",
    re.DOTALL,
)


def escape_xml_prose(text: str, prose_tags: Collection[str]) -> str:
    """Escape bare punctuation in prose, leaving markup and entities to validation."""
    prose_depth = 0

    def escape_prose(match: re.Match[str]) -> str:
        nonlocal prose_depth
        token = match.group(0)
        if match.lastgroup == "tag":
            name = token.lstrip("</").split(None, 1)[0].rstrip("/>")
            if name in prose_tags:
                prose_depth += -1 if token.startswith("</") else 0 if token.endswith("/>") else 1
        elif match.lastgroup == "bare" and prose_depth > 0:
            if token == "&":
                return "&amp;"
            if text[match.end():match.end() + 1] not in {"/", "!", "?"}:
                return "&lt;"
        return token

    return XML_TOKEN.sub(escape_prose, text)
