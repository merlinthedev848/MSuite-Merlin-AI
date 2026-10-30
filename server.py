"""
Backward-compatibility entrypoint: delegates to the unified FastAPI application in main.py.
Eliminates the 827-line duplicate SimpleHTTPRequestHandler server monolith.
"""
from main import HOST, PORT, app, format_bytes, get_engine, run  # noqa: F401

if __name__ == "__main__":
    run()
