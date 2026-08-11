"""Start the read-only Machanize Phase 3 API."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "machanize.phase3.api:create_app",
        host="127.0.0.1",
        port=8000,
        # Background Gemini/YOLO jobs are process-local. Reloading would abandon them.
        reload=os.environ.get("MACHANIZE_DEV_RELOAD") == "1",
        factory=True,
    )
