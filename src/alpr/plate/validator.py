"""Tunisian plate text normalization and syntax validation.

Reading order is the plate's visual order, left to right, as in the labelled data:

* Ordinary plate:  ``117 تونس 3989``  -> series (1-3 digits), the word تونس, serial (1-4 digits).
  Compact label form ``117T3989``; canonical text ``117 TUN 3989``.
* Temporary / special-regime plate (RS):  ``103679 ن ت``  -> 5 or 6 digits then ``ن ت``.
  Compact label form ``103679N``; canonical text ``103679 NT``.

Rules:

* Only fix characters that are unambiguous look-alikes of a digit (``O`` -> ``0``, ``I`` -> ``1``,
  Arabic-Indic digits -> ASCII). Never add, drop or guess digits to make a plate valid.
* Anything that does not match a known format is returned as ``INVALID_FORMAT`` with the cleaned
  text, so the UI can still show what was read.

Other Tunisian formats (government, diplomatic CD/CMD, dealer, military) are out of scope for now:
they are rare in the datasets and are reported as ``INVALID_FORMAT``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alpr.contracts import PlateStatus

# Compact alphabet used by the OCR model and the labels: digits, T (= تونس), N (= ن ت).
OCR_ALPHABET = "0123456789TN"

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Letters that OCR engines output for a digit. Applied only to characters outside T/N markers.
_DIGIT_LOOKALIKES = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1",
                                   "Z": "2", "S": "5", "B": "8", "G": "6"})
_TUNIS_WORD = re.compile(r"(تونس|تو\s*نس|TUNIS|TUN|TN)", re.IGNORECASE)
_NT_WORD = re.compile(r"(ن\s*ت|ت\s*ن|NT|RS)", re.IGNORECASE)

ORDINARY = re.compile(r"(\d{1,3})T(\d{1,4})")
REGIME_SUSPENSIF = re.compile(r"(\d{5,6})N")  # 5-6 digits; all RS plates in the data have 6


@dataclass(frozen=True)
class Validation:
    text: str  # canonical text, or the cleaned compact text when invalid
    status: PlateStatus
    plate_format: str | None  # "ordinary", "rs" or None


def to_compact(raw: str) -> str:
    """Map any OCR output to the compact alphabet ``0-9 T N`` (other characters dropped)."""
    text = raw.translate(_ARABIC_INDIC)
    text = _TUNIS_WORD.sub("T", text)
    text = _NT_WORD.sub("N", text)
    text = text.upper()
    out = []
    for ch in text:
        if ch in OCR_ALPHABET:
            out.append(ch)
        elif ch in "OQDILZSBG|":
            out.append(ch.translate(_DIGIT_LOOKALIKES))
        # spaces, dots, dashes, stray Arabic letters: dropped
    return "".join(out)


def validate(raw: str) -> Validation:
    compact = to_compact(raw)
    if not any(ch.isdigit() for ch in compact):
        return Validation(text="", status=PlateStatus.NO_TEXT, plate_format=None)
    if m := ORDINARY.fullmatch(compact):
        return Validation(f"{m.group(1)} TUN {m.group(2)}", PlateStatus.VALID, "ordinary")
    if m := REGIME_SUSPENSIF.fullmatch(compact):
        return Validation(f"{m.group(1)} NT", PlateStatus.VALID, "rs")
    return Validation(compact, PlateStatus.INVALID_FORMAT, None)


def canonical_to_compact(text: str) -> str:
    """``117 TUN 3989`` -> ``117T3989``; used to compare predictions with dataset labels."""
    return to_compact(text)
