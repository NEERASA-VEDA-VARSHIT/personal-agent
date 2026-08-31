# Backward-compat shim — use app.models.providers.openai
from app.models.providers.openai import *  # noqa: F401,F403
from app.models.providers.openai import OpenAICompatibleProvider  # noqa: F401