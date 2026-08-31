import sys
from pathlib import Path

# Ensure src and root are importable when run as `uvicorn apps.api.main:app`
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI

try:
    from apps.api.routes.chat import router as chat_router
except ImportError:
    from routes.chat import router as chat_router  # fallback when pythonpath is apps/api

app = FastAPI(title="Personal Agent API", version="0.1.0")
app.include_router(chat_router)

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
