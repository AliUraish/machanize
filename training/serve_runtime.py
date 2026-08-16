"""Start the backend-only Machanize runtime monitoring API."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "machanize.runtime.api:create_runtime_app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        factory=True,
    )
