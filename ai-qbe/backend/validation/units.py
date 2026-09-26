"""Lightweight physics-unit presence/plausibility checking (docs/PHASE0-DESIGN.md
section 13: "a lightweight unit-consistency checker (regex/lookup against
a small SI-unit table)").

Deliberately narrow scope, matching the Phase 0 design: this flags a
numeric option that looks like it's missing a unit it should have, and
recognizes the common SI/derived units so a legitimately-unitful option
isn't mistaken for a bare unitless number. It does NOT attempt full
dimensional analysis (e.g. verifying that adding two options gives a
dimensionally consistent result) -- that's out of scope for this phase's
"lightweight" checker, and a false sense of rigor there would be worse
than the honest, narrow check this module actually performs.
"""

from __future__ import annotations

import re

# SI base + common derived units used in undergraduate physics, plus
# their standard symbols. Prefixed forms (km, ms, kJ, ...) are handled by
# the regex allowing an optional single/double-letter SI prefix.
_SI_UNIT_SYMBOLS = {
    "m", "kg", "s", "A", "K", "mol", "cd",  # base units
    "N", "J", "W", "Pa", "Hz", "C", "V", "Ω", "ohm", "F", "T", "Wb", "H",
    "lm", "lx", "Bq", "Gy", "Sv", "kat",
    "m/s", "m/s^2", "m/s²", "kg/m^3", "kg/m³",
    "rad", "sr", "°", "deg", "%", "mol/L", "M",
}
_SI_PREFIXES = "|".join(["k", "M", "G", "T", "m", "µ", "u", "n", "p", "c", "d", "h"])

_UNIT_TOKEN_RE = re.compile(
    r"\b(?:" + _SI_PREFIXES + r")?(" + "|".join(re.escape(u) for u in sorted(_SI_UNIT_SYMBOLS, key=len, reverse=True)) + r")\b"
)
_BARE_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?(\s*[eE][+-]?\d+)?$")
_NUMBER_WITH_TRAILING_TEXT_RE = re.compile(r"^-?\d+(\.\d+)?(\s*[eE][+-]?\d+)?\s*(.*)$")


def looks_numeric(text: str) -> bool:
    return bool(_BARE_NUMBER_RE.match(text.strip()))


_SYMBOL_ONLY_UNITS = {"°", "%", "Ω"}


def has_recognized_unit(text: str) -> bool:
    stripped = text.strip()
    if stripped in _SYMBOL_ONLY_UNITS or any(stripped.endswith(u) for u in _SYMBOL_ONLY_UNITS):
        return True
    return bool(_UNIT_TOKEN_RE.search(text))


def check_numeric_option_has_unit(text: str) -> bool | None:
    """Returns True if the option is numeric and carries a recognized
    unit, False if it's numeric but appears to carry no unit at all
    (just trailing non-unit text or nothing), or None if the option isn't
    a numeric value in the first place (unit-checking doesn't apply, e.g.
    a conceptual multiple-choice option like "Newton's second law").
    """
    text = text.strip()
    match = _NUMBER_WITH_TRAILING_TEXT_RE.match(text)
    if not match:
        return None  # not a numeric-leading option at all

    trailing = match.group(3).strip()
    if not trailing:
        return False  # bare number, no unit
    return has_recognized_unit(trailing)
