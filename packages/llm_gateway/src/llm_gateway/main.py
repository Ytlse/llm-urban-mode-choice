"""
main.py — uvicorn entry point (docker-compose compat: `llm_gateway.main:app`).

The application is built by the create_app() factory (api/app.py):
explicit dependency composition, RPM reset in the lifespan.
"""

from llm_gateway.api.app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_gateway.main:app", host="0.0.0.0", port=8000, reload=True)
