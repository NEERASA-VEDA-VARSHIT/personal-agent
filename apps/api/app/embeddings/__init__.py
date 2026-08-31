# Backward-compat shim — use app.models.embeddings
from app.models.embeddings import *  # noqa: F401,F403
from app.models.embeddings import EmbeddingService, cosine_similarity  # noqa: F401