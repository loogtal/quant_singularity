"""Tests for config helpers."""

import os
import pytest

from config.settings import _env_bool, _env_float, _env_int


class TestEnvBool:
    def test_true_values(self):
        for v in ("true", "1", "yes", "TRUE", "True"):
            os.environ["_TEST_BOOL"] = v
            assert _env_bool("_TEST_BOOL", False) is True

    def test_false_values(self):
        for v in ("false", "0", "no"):
            os.environ["_TEST_BOOL"] = v
            assert _env_bool("_TEST_BOOL", True) is False

    def test_default(self):
        os.environ.pop("_TEST_MISSING", None)
        assert _env_bool("_TEST_MISSING", True) is True

    def test_empty_string(self):
        os.environ["_TEST_BOOL"] = ""
        result = _env_bool("_TEST_BOOL", True)
        assert isinstance(result, bool)


class TestEnvFloat:
    def test_valid(self):
        os.environ["_TEST_F"] = "3.14"
        assert _env_float("_TEST_F", 0) == pytest.approx(3.14)

    def test_default(self):
        os.environ.pop("_TEST_MISSING", None)
        assert _env_float("_TEST_MISSING", 1.5) == 1.5


class TestEnvInt:
    def test_valid(self):
        os.environ["_TEST_I"] = "42"
        assert _env_int("_TEST_I", 0) == 42

    def test_default(self):
        os.environ.pop("_TEST_MISSING", None)
        assert _env_int("_TEST_MISSING", 10) == 10
