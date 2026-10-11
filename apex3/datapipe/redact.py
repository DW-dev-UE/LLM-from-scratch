from __future__ import annotations

import ipaddress
import re

SECRETS = [("private_key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"), "<PRIVATE_KEY>", ("PRIVATE KEY",)), ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<SECRET>", ("AKIA", "ASIA")), ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"), "<SECRET>", ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")),
           ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"), "<SECRET>", ("xox",)), ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "<SECRET>", ("AIza",)), ("stripe_key", re.compile(r"\b[rs]k_(?:live|test)_[0-9A-Za-z]{16,}\b"), "<SECRET>", ("k_live_", "k_test_")), ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{32,}\b"), "<SECRET>", ("sk-",)), ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "<SECRET>", ("eyJ",))]
_ASSIGN = re.compile(r"(?i)(\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)\b\s*[:=]\s*)(['\"])([^'\"\s]{8,})\2")
_ASSIGN_KEYS = ("password", "passwd", "pwd", "secret", "apikey", "api_key", "api-key", "accesstoken", "access_token", "access-token", "authtoken", "auth_token", "auth-token")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@([A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,})\b")
_EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net", "localhost", "test.com")
_NUMERIC = re.compile(r"(?P<rrn>(?<!\d)\d{6}-[1-4]\d{6}(?!\d))|(?P<phone>(?<!\d)01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}(?!\d))|(?P<ip>(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.]))")


def redact(text: str, secrets: bool = True, pii: bool = True) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def bump(name: str, n: int = 1) -> None:
        if n:
            counts[name] = counts.get(name, 0) + n

    if secrets:
        for name, pattern, placeholder, literals in SECRETS:
            if any(lit in text for lit in literals):
                text, n = pattern.subn(placeholder, text)
                bump(name, n)
        low = text.lower()
        if any(k in low for k in _ASSIGN_KEYS):
            text, n = _ASSIGN.subn(r"\1\2<SECRET>\2", text)
            bump("assignment", n)
    if pii:
        def email(m):
            if m.group(1).lower().endswith(_EXAMPLE_DOMAINS):
                return m.group(0)
            bump("email")
            return "<EMAIL>"

        def numeric(m):
            if m.group("rrn"):
                bump("kr_rrn")
                return "<ID_NUMBER>"
            if m.group("phone"):
                bump("kr_phone")
                return "<PHONE>"
            try:
                if not ipaddress.ip_address(m.group(0)).is_global:
                    return m.group(0)
            except ValueError:
                return m.group(0)
            bump("ip")
            return "<IP_ADDRESS>"

        if "@" in text:
            text = _EMAIL.sub(email, text)
        text = _NUMERIC.sub(numeric, text)
    return text, counts
