"""FastAPI app entrypoint."""
from fastapi import FastAPI

app = FastAPI(title="rag_app")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check."""
    return {"status": "ok"}
