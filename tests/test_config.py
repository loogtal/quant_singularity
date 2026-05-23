"""Tests for config/settings.py helper functions."""

import os

from config.settings import _env_bool, _env_float, _env_int


class TestEnvBool:
    def test_true_values(self):
        for val in ("1", "true", "yes", "on", "True", "YES", "ON"):
            os.environ["_TEST_BOOL"] = val
            assert _env_bool("_TEST_BOOL") is True
            del os.environ["_TEST_BOOL"]

    def test_false_values(self):
        for val in ("0", "false", "no", "off", "random"):
            os.environ["_TEST_BOOL"] = val
            assert _env_bool("_TEST_BOOL") is False
            del os.environ["_TEST_BOOL"]

    def test_default(self):
        assert _env_bool("_NONEXISTENT_KEY_XYZ", False) is False
        assert _env_bool("_NONEXISTENT_KEY_XYZ", True) is True


class TestEnvFloat:
    def test_valid(self):
        os.environ["_TEST_FLOAT"] = "3.14"
        assert _env_float("_TEST_FLOAT", 0.0) == 3.14
        del os.environ["_TEST_FLOAT"]

    def test_invalid(self):
        os.environ["_TEST_FLOAT"] = "abc"
        assert _env_float("_TEST_FLOAT", 1.5) == 1.5
        del os.environ["_TEST_FLOAT"]

    def test_default(self):
        assert _env_float("_NONEXISTENT_KEY_XYZ", 9.9) == 9.9


class TestEnvInt:
    def test_valid(self):
        os.environ["_TEST_INT"] = "42"
        assert _env_int("_TEST_INT", 0) == 42
        del os.environ["_TEST_INT"]

    def test_invalid(self):
        os.environ["_TEST_INT"] = "not_a_number"
        assert _env_int("_TEST_INT", 10) == 10
        del os.environ["_TEST_INT"]

    def test_default(self):
        assert _env_int("_NONEXISTENT_KEY_XYZ", 7) == 7
