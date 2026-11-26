import pytest

from electrolyte_screening.data import binary_label


@pytest.mark.parametrize(
    ("reading", "expected"),
    [
        ("0", 0),
        ("9.99999e-5", 0),
        ("1e-4", 1),
        ("0.00010000001", 1),
        ("2E-4", 1),
        ("<1E-10", 0),
        ("  < 1E-4 ", 0),
        ("<=9e-5", 0),
        ("<=1e-4", None),
        ("≤1e-4", None),
        ("<1e-3", None),
        ("", None),
    ],
)
def test_threshold_and_upper_bounds(reading, expected):
    assert binary_label(reading) == expected


@pytest.mark.parametrize("reading", ["nan", "-1e-5", "reported elsewhere"])
def test_invalid_conductivity_fails_explicitly(reading):
    with pytest.raises(ValueError, match="Invalid conductivity"):
        binary_label(reading)
