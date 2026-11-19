"""
Unit Tests for Indian Plate Validator & Text Normalizer
======================================================
"""

import pytest
from app.recognition.plate_validator import IndianVehicleNumberValidator, ValidationResult
from app.recognition.text_normalizer import TextNormalizer


def test_text_normalizer_clean():
    assert TextNormalizer.clean("KA 01 AB 1234") == "KA01AB1234"
    assert TextNormalizer.clean("ka-01-ab-1234") == "KA01AB1234"
    assert TextNormalizer.clean("  DL.3C.1234\n") == "DL3C1234"
    assert TextNormalizer.clean("") == ""
    assert TextNormalizer.clean(None) == ""


def test_standard_valid_plates():
    validator = IndianVehicleNumberValidator()

    # Karnataka standard
    res = validator.validate("KA01AB1234")
    assert res.is_valid is True
    assert res.normalized_plate == "KA01AB1234"
    assert res.format_type == "STANDARD"
    assert res.state_code == "KA"
    assert "Karnataka" in res.state_name

    # Maharashtra single letter series
    res2 = validator.validate("MH12A5678")
    assert res2.is_valid is True
    assert res2.normalized_plate == "MH12A5678"
    assert res2.state_code == "MH"

    # Delhi with district & series
    res3 = validator.validate("DL3CAB9999")
    assert res3.is_valid is True
    assert res3.state_code == "DL"

    # Legacy format without series letters
    res4 = validator.validate("KA041234")
    assert res4.is_valid is True
    assert res4.state_code == "KA"


def test_bharat_series_plates():
    validator = IndianVehicleNumberValidator()

    res = validator.validate("22BH1234AA")
    assert res.is_valid is True
    assert res.normalized_plate == "22BH1234AA"
    assert res.format_type == "BHARAT_SERIES"
    assert res.state_code == "BH"

    res2 = validator.validate("21BH6543A")
    assert res2.is_valid is True
    assert res2.format_type == "BHARAT_SERIES"


def test_positional_character_substitution():
    validator = IndianVehicleNumberValidator()

    # 'O' and 'I' in district position where digits expected -> '0' and '1'
    res = validator.validate("KAOIAB1234")
    assert res.is_valid is True
    assert res.normalized_plate == "KA01AB1234"
    assert res.confidence_penalty > 0.0

    # '8' in series position where letters expected -> 'B'
    res2 = validator.validate("KA01A81234")
    assert res2.is_valid is True
    assert res2.normalized_plate == "KA01AB1234"

    # 'O' in number position where digits expected -> '0'
    res3 = validator.validate("KA01AB123O")
    assert res3.is_valid is True
    assert res3.normalized_plate == "KA01AB1230"

    # Bharat series: 'I' in 4-digit number where digit expected -> '1'
    res4 = validator.validate("22BHI234AA")
    assert res4.is_valid is True
    assert res4.normalized_plate == "22BH1234AA"


def test_invalid_plates_rejected():
    validator = IndianVehicleNumberValidator()

    # Random text
    assert validator.validate("HELLO WORLD").is_valid is False
    # Non-existent state code
    assert validator.validate("ZZ99AB1234").is_valid is False
    # Incomplete plate
    assert validator.validate("KA01").is_valid is False
    # Numbers only
    assert validator.validate("1234567890").is_valid is False
    # Empty string
    assert validator.validate("").is_valid is False
