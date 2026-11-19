"""
Text Normalization for License Plates
=====================================
Cleans raw OCR strings by stripping whitespace, hyphens, periods,
and unwanted characters before validation.
"""

import re
from typing import Tuple


class TextNormalizer:
    # Character removal pattern: keep only ASCII alphanumeric
    CLEAN_PATTERN = re.compile(r"[^A-Za-z0-9]")

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Normalize text to uppercase alphanumeric string.
        Strips spaces, newlines, tabs, hyphens, dots, and symbols.
        """
        if not text:
            return ""
        # Remove common OCR artifacts like leading/trailing single quotes or noise
        cleaned = cls.CLEAN_PATTERN.sub("", text)
        return cleaned.upper().strip()

    @classmethod
    def decompose(cls, text: str) -> Tuple[str, str]:
        """
        Returns (original_trimmed, cleaned_alphanumeric).
        """
        raw = text.strip() if text else ""
        cleaned = cls.clean(raw)
        return raw, cleaned
