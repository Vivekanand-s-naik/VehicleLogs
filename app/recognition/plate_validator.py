"""
Indian Vehicle Registration Number Validator
============================================
Validates and repairs OCR output against Indian registration formats using
*template matching with a cost model* instead of blind global replacement.

Supported formats
-----------------
- Standard:      SS DD L(L)(L) NNNN   e.g. KA01AB1234, MH12A1234, KA041234
- Delhi:         DL D(D) L(LL) NNNN   e.g. DL3CAB1234, DL10C1234
- Bharat Series: YY BH NNNN L(L)      e.g. 22BH1234AA

How repair works
----------------
1. Text is normalized to A-Z0-9.
2. Every substring window of plausible length is tried (removes OCR junk such
   as a leading "IND" or a trailing border artifact).
3. For each window every format template (district length x series length) is
   tried. A character only changes when the template position demands the
   other class (digit vs letter) AND the swap is a known OCR confusion
   (O->0, I->1, B->8, S->5, Z->2, G->6, A->4, T->7, D->0 ... and reverse).
4. The state code must be a real RTO state/UT code. A few letter-letter
   confusions (M/N, O/D/Q, U/V, C/G) are allowed ONLY for the state code.
5. The lowest-cost candidate wins. Cost becomes `confidence_penalty`, which
   the OCR stage subtracts from the OCR confidence, so heavily repaired reads
   are less likely to be accepted.

Regex validity does NOT prove a vehicle is officially registered.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from app.recognition.text_normalizer import TextNormalizer


@dataclass
class ValidationResult:
    is_valid: bool
    normalized_plate: str
    raw_text: str
    format_type: str  # "STANDARD", "BHARAT_SERIES", or "INVALID"
    state_code: Optional[str] = None
    state_name: Optional[str] = None
    confidence_penalty: float = 0.0
    substitutions: int = 0
    error_message: Optional[str] = None


class IndianVehicleNumberValidator:
    INDIAN_STATES: Dict[str, str] = {
        "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
        "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
        "DD": "Daman and Diu", "DL": "Delhi", "DN": "Dadra and Nagar Haveli", "GA": "Goa",
        "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand",
        "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh",
        "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya", "MN": "Manipur",
        "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha",
        "OR": "Odisha (Legacy)", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan",
        "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana",
        "UK": "Uttarakhand", "UA": "Uttarakhand (Legacy)", "UP": "Uttar Pradesh", "WB": "West Bengal",
    }

    # Digit -> letter (used where the template demands a letter)
    TO_LETTERS: Dict[str, str] = {
        "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B",
    }
    # Letter -> digit (strict visual lookalikes only; avoids twisting real letters into digits)
    TO_DIGITS: Dict[str, str] = {
        "O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7",
    }
    # Letter <-> letter confusions, allowed ONLY inside the 2-letter state code
    STATE_LETTER_CONFUSIONS: Dict[str, str] = {
        "M": "N", "N": "M", "O": "D", "D": "O", "Q": "O", "U": "V", "V": "U", "C": "G", "G": "C",
        "H": "N", "W": "M",
    }

    # Cost model
    SUB_COST = 0.05          # digit<->letter swap
    BOUNDARY_SUB_COST = 0.14 # coercing a letter into the 1st digit of the 4-digit number (often a truncated plate!)
    STATE_FIX_COST = 0.08    # letter<->letter swap inside the state code
    TRIM_COST = 0.06         # per junk character removed from the ends (> SUB_COST: prefer repair over deletion)
    IND_TRIM_COST = 0.01     # removing a leading "IND" (the plate's hologram/IND strip) is nearly free
    SERIES_IO_COST = 0.05    # 'I'/'O' are not issued in Indian series letters

    MIN_LEN, MAX_LEN = 8, 11
    MAX_INPUT_LEN = 18

    # ------------------------------------------------------------------ helpers
    @classmethod
    def _coerce(cls, ch: str, want: str) -> Tuple[Optional[str], int]:
        """Return (char, substitutions) coerced to 'A' (letter) or 'D' (digit), or (None, 0)."""
        if want == "A":
            if ch.isalpha():
                return ch, 0
            sub = cls.TO_LETTERS.get(ch)
            return (sub, 1) if sub else (None, 0)
        if ch.isdigit():
            return ch, 0
        sub = cls.TO_DIGITS.get(ch)
        return (sub, 1) if sub else (None, 0)

    @classmethod
    def _coerce_run(cls, text: str, want: str) -> Tuple[Optional[str], int]:
        out, subs = [], 0
        for ch in text:
            c, s = cls._coerce(ch, want)
            if c is None:
                return None, 0
            out.append(c)
            subs += s
        return "".join(out), subs

    @classmethod
    def _resolve_state(cls, raw2: str) -> Tuple[Optional[str], float, int]:
        """Resolve the 2-char state code. Returns (code, cost, substitutions)."""
        letters, subs = cls._coerce_run(raw2, "A")
        if letters is None:
            return None, 0.0, 0
        if letters in cls.INDIAN_STATES:
            return letters, subs * cls.SUB_COST, subs
        best: Optional[Tuple[str, float, int]] = None
        for i in range(2):
            alt = cls.STATE_LETTER_CONFUSIONS.get(letters[i])
            if not alt:
                continue
            cand = letters[:i] + alt + letters[i + 1:]
            if cand in cls.INDIAN_STATES:
                cost = subs * cls.SUB_COST + cls.STATE_FIX_COST
                if best is None or cost < best[1]:
                    best = (cand, cost, subs + 1)
        return best if best else (None, 0.0, 0)

    # ---------------------------------------------------------------- matchers
    @classmethod
    def _match_standard(cls, w: str) -> Optional[Tuple[str, float, int, str]]:
        """Best standard-format interpretation of window `w` (no trimming)."""
        n = len(w)
        state, s_cost, s_subs = cls._resolve_state(w[:2])
        if state is None:
            return None
        number, n_subs = cls._coerce_run(w[-4:], "D")
        if number is None or number == "0000":
            return None
        # Coercing a non-O/I letter (e.g. G/B/S/Z) at the start of the 4-digit number
        # is frequently a truncated 10-char plate whose 2nd series letter bled into slot -4.
        boundary_letter = w[-4].isalpha() and w[-4] not in ("O", "I")
        middle = w[2:-4]
        dist_lengths = (1, 2) if state == "DL" else (2,)
        max_series = 3 if state == "DL" else 2

        best = None
        for d_len in dist_lengths:
            s_len = len(middle) - d_len
            if s_len < 0 or s_len > max_series:
                continue
            # Never fabricate a rare 0-series legacy plate by coercing a letter into a digit
            if s_len == 0 and n_subs > 0:
                continue
            district, d_subs = cls._coerce_run(middle[:d_len], "D")
            if district is None or int(district) == 0:
                continue
            series, se_subs = cls._coerce_run(middle[d_len:], "A")
            if series is None:
                continue
            subs = n_subs + d_subs + se_subs
            cost = s_cost + subs * cls.SUB_COST
            if boundary_letter and s_len == 1:
                cost += cls.BOUNDARY_SUB_COST
            cost += sum(cls.SERIES_IO_COST for c in series if c in "IO")
            cand = f"{state}{district}{series}{number}"
            if best is None or cost < best[1]:
                best = (cand, cost, subs + s_subs, state)
        return best

    @classmethod
    def _match_bharat(cls, w: str) -> Optional[Tuple[str, float, int, str]]:
        if len(w) not in (9, 10):
            return None
        year, y_subs = cls._coerce_run(w[:2], "D")
        bh, b_subs = cls._coerce_run(w[2:4], "A")
        number, n_subs = cls._coerce_run(w[4:8], "D")
        series, s_subs = cls._coerce_run(w[8:], "A")
        if None in (year, bh, number, series) or bh != "BH":
            return None
        subs = y_subs + b_subs + n_subs + s_subs
        cost = subs * cls.SUB_COST + sum(cls.SERIES_IO_COST for c in series if c in "IO")
        return f"{year}BH{number}{series}", cost, subs, "BH"

    # ------------------------------------------------------------------ public
    @classmethod
    def candidates(cls, cleaned: str) -> List[Tuple[str, float, int, str, str]]:
        """All valid interpretations: (plate, cost, substitutions, state_code, format)."""
        out = []
        n = len(cleaned)
        for length in range(min(cls.MAX_LEN, n), cls.MIN_LEN - 1, -1):
            for start in range(0, n - length + 1):
                w = cleaned[start:start + length]
                trimmed = n - length
                if start >= 3 and cleaned.startswith("IND"):
                    trim = cls.IND_TRIM_COST + (trimmed - 3) * cls.TRIM_COST
                else:
                    trim = trimmed * cls.TRIM_COST
                std = cls._match_standard(w)
                if std:
                    out.append((std[0], std[1] + trim, std[2], std[3], "STANDARD"))
                bh = cls._match_bharat(w)
                if bh:
                    out.append((bh[0], bh[1] + trim, bh[2], bh[3], "BHARAT_SERIES"))
        out.sort(key=lambda c: (c[1], -len(c[0])))
        return out

    @classmethod
    def validate(cls, raw_text: str) -> ValidationResult:
        if not raw_text:
            return ValidationResult(False, "", "", "INVALID", error_message="Empty input text")

        cleaned = TextNormalizer.clean(raw_text)
        if len(cleaned) < cls.MIN_LEN or len(cleaned) > cls.MAX_INPUT_LEN:
            return ValidationResult(
                False, cleaned, raw_text, "INVALID",
                error_message=f"Invalid plate character length ({len(cleaned)})",
            )

        cands = cls.candidates(cleaned)
        if not cands:
            return ValidationResult(
                False, cleaned, raw_text, "INVALID",
                error_message="Text does not match any recognized Indian registration format",
            )

        plate, cost, subs, state, fmt = cands[0]
        state_name = "Bharat Series (Pan-India)" if fmt == "BHARAT_SERIES" else cls.INDIAN_STATES.get(state)
        return ValidationResult(
            is_valid=True,
            normalized_plate=plate,
            raw_text=raw_text,
            format_type=fmt,
            state_code=state,
            state_name=state_name,
            confidence_penalty=round(cost, 4),
            substitutions=subs,
        )
