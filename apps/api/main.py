from fastapi import FastAPI

app = FastAPI(title=""Personal Agent API"", version=""0.1.0"")

@app.get(""/health"")
def health():
    return {""status"": ""ok"", ""version"": ""0.1.0""}

# Routes will be included here:
# from apps.api.routes import chat, memories, decisions
