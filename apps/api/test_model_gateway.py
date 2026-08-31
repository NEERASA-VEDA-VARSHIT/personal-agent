import unittest

from app.models.config import ModelSettings
from app.models.gateway import ModelGateway, build_provider, get_default_gateway
from app.models.openai_provider import OpenAICompatibleProvider


class TestModelGateway(unittest.TestCase):
    def test_default_gateway_uses_openai_compatible_provider(self) -> None:
        gateway = get_default_gateway()
        self.assertIsInstance(gateway.provider, OpenAICompatibleProvider)
        self.assertEqual(gateway.provider.default_model, "llama3.2")

    def test_build_provider_uses_settings_provider_name(self) -> None:
        settings = ModelSettings(provider="ollama", model="llama3.2", base_url="http://localhost:11434/v1", api_key="ollama")
        provider = build_provider(settings)
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertEqual(provider.default_model, "llama3.2")

    def test_gateway_accepts_explicit_provider(self) -> None:
        provider = build_provider(ModelSettings(provider="ollama", model="llama3.2"))
        gateway = ModelGateway(provider=provider)
        self.assertIs(gateway.provider, provider)


if __name__ == "__main__":
    unittest.main()
