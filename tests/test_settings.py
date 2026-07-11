"""Unit tests for Settings loading and configure()."""
import os
import unittest
from unittest import mock

from listening_ai.settings import Settings, configure, get_settings


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        s = Settings()
        self.assertEqual(s.store_backend, "json")
        self.assertTrue(s.openrouter_api_url.startswith("https://"))
        self.assertEqual(s.token_expiry_seconds, 86400 * 7)

    def test_from_mapping_snake_and_caps(self):
        s = Settings.from_mapping({
            "store_backend": "spaces",
            "OPENROUTER_API_KEY": "sk-test",
            "DO_SPACES_BUCKET": "my-bucket",
            "S3_PREFIX": "app/listening/",
        })
        self.assertEqual(s.store_backend, "spaces")
        self.assertEqual(s.openrouter_api_key, "sk-test")
        self.assertEqual(s.spaces_bucket, "my-bucket")
        self.assertEqual(s.spaces_prefix, "app/listening/")

    def test_from_config_module(self):
        class FakeConfig:
            OPENROUTER_API_KEY = "from-module"
            STORE_BACKEND = "json"
            LLM_TEMPERATURE = 0.2

        s = Settings.from_config_module(FakeConfig)
        self.assertEqual(s.openrouter_api_key, "from-module")
        self.assertEqual(s.store_backend, "json")
        self.assertEqual(s.llm_temperature, 0.2)

    def test_from_env(self):
        env = {
            "OPENROUTER_API_KEY": "env-key",
            "LISTENING_AI_STORE": "spaces",
            "LISTENING_AI_DATA_DIR": "/tmp/listening-test",
            "LISTENING_AI_PREFIX": "pfx/",
            "DO_SPACES_KEY": "k",
            "DO_SPACES_SECRET": "s",
            "DO_SPACES_BUCKET": "b",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            s = Settings.from_env()
        self.assertEqual(s.openrouter_api_key, "env-key")
        self.assertEqual(s.store_backend, "spaces")
        self.assertEqual(s.data_dir, "/tmp/listening-test")
        self.assertEqual(s.spaces_prefix, "pfx/")
        self.assertEqual(s.spaces_bucket, "b")

    def test_resolved_db_path(self):
        s = Settings(data_dir="/data", db_path=None)
        self.assertEqual(s.resolved_db_path(), os.path.join("/data", "db.json"))
        s2 = Settings(db_path="/explicit/db.json")
        self.assertEqual(s2.resolved_db_path(), "/explicit/db.json")

    def test_resolved_spaces_endpoint(self):
        s = Settings(spaces_region="nyc3", spaces_endpoint="")
        self.assertEqual(s.resolved_spaces_endpoint(), "https://nyc3.digitaloceanspaces.com")
        s2 = Settings(spaces_endpoint="https://custom.example")
        self.assertEqual(s2.resolved_spaces_endpoint(), "https://custom.example")

    def test_configure_overrides(self):
        configure(Settings(openrouter_api_key="a"), openrouter_api_key="b", store_backend="json")
        s = get_settings()
        self.assertEqual(s.openrouter_api_key, "b")
        self.assertEqual(s.store_backend, "json")


if __name__ == "__main__":
    unittest.main()
