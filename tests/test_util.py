"""Unit tests for shared utilities."""
import unittest

from listening_ai.util import utc_now_iso


class UtilTests(unittest.TestCase):
    def test_utc_now_iso_format(self):
        s = utc_now_iso()
        self.assertTrue(s.endswith("Z"))
        self.assertRegex(s, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


if __name__ == "__main__":
    unittest.main()
