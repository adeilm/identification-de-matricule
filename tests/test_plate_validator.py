import pytest

from alpr.contracts import PlateStatus
from alpr.plate.validator import to_compact, validate


@pytest.mark.parametrize(
    "raw, text, fmt",
    [
        ("117T3989", "117 TUN 3989", "ordinary"),  # dataset label form
        ("117 تونس 3989", "117 TUN 3989", "ordinary"),  # as printed on the plate
        ("94 TN 3458", "94 TUN 3458", "ordinary"),
        ("121T237", "121 TUN 237", "ordinary"),  # short serials exist
        ("103679N", "103679 NT", "rs"),
        ("103679 ن ت", "103679 NT", "rs"),
        ("١١٧ تونس ٣٩٨٩", "117 TUN 3989", "ordinary"),  # Arabic-Indic digits
    ],
)
def test_valid_formats(raw, text, fmt):
    result = validate(raw)
    assert result.status is PlateStatus.VALID
    assert result.text == text
    assert result.plate_format == fmt


def test_digit_lookalikes_are_fixed():
    assert validate("1I7 TUN 39B9").text == "117 TUN 3989"


@pytest.mark.parametrize("raw", ["1234T3989", "117T39890", "117", "T3989", "1179TN", "5948N"])
def test_wrong_digit_counts_are_invalid_not_repaired(raw):
    result = validate(raw)
    assert result.status is PlateStatus.INVALID_FORMAT
    assert result.text == to_compact(raw)  # nothing added or removed


@pytest.mark.parametrize("raw", ["", "   ", "تونس", "--"])
def test_no_digits_means_no_text(raw):
    assert validate(raw).status is PlateStatus.NO_TEXT
