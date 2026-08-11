"""Start the read-only Machanize Phase 3 API."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "machanize.phase3.api:create_app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        factory=True,
    )
