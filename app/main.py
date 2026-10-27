from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.routes import router


# ============================================================
# PATHS
# ============================================================

# Project root:
# RAG_Chatbot/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Streamlit:
# RAG_Chatbot/app/ui/streamlit_app.py
STREAMLIT_APP = (
    PROJECT_ROOT
    / "app"
    / "ui"
    / "streamlit_app.py"
)


# ============================================================
# SERVER CONFIGURATION
# ============================================================

FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000

STREAMLIT_HOST = "127.0.0.1"
STREAMLIT_PORT = 8501


# Streamlit process started by this application.
streamlit_process: Optional[subprocess.Popen] = None

# True only if THIS FastAPI process started Streamlit.
streamlit_started_by_us = False


# ============================================================
# PORT CHECK
# ============================================================

def is_port_open(
    host: str,
    port: int,
) -> bool:
    """
    Check whether a TCP port is accepting connections.
    """

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as sock:

        sock.settimeout(0.5)

        return (
            sock.connect_ex(
                (host, port)
            )
            == 0
        )


# ============================================================
# WAIT FOR STREAMLIT
# ============================================================

async def wait_for_streamlit(
    timeout: float = 20.0,
) -> bool:
    """
    Wait until Streamlit starts listening.
    """

    elapsed = 0.0

    while elapsed < timeout:

        if is_port_open(
            STREAMLIT_HOST,
            STREAMLIT_PORT,
        ):
            return True

        await asyncio.sleep(0.25)
        elapsed += 0.25

    return False


# ============================================================
# START STREAMLIT
# ============================================================

def start_streamlit() -> None:
    """
    Start Streamlit automatically.

    Streamlit runs internally on port 8501.
    Users don't need to open that port directly.
    """

    global streamlit_process
    global streamlit_started_by_us

    # --------------------------------------------------------
    # Verify Streamlit application exists.
    # --------------------------------------------------------

    if not STREAMLIT_APP.exists():

        raise FileNotFoundError(
            "\n"
            "Streamlit application not found.\n\n"
            f"Expected:\n"
            f"{STREAMLIT_APP}\n\n"
            "Make sure the file exists at:\n"
            "app/ui/streamlit_app.py"
        )

    # --------------------------------------------------------
    # Check whether Streamlit is already running.
    # --------------------------------------------------------

    if is_port_open(
        STREAMLIT_HOST,
        STREAMLIT_PORT,
    ):

        print(
            "Streamlit is already running on "
            f"http://{STREAMLIT_HOST}:{STREAMLIT_PORT}"
        )

        streamlit_started_by_us = False

        return

    # --------------------------------------------------------
    # Environment variables for Streamlit.
    # --------------------------------------------------------

    env = os.environ.copy()

    # Streamlit frontend talks to FastAPI.
    env["NEXA_API_URL"] = (
        f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"
    )

    # --------------------------------------------------------
    # Streamlit command.
    # --------------------------------------------------------

    command = [
        sys.executable,

        "-m",
        "streamlit",

        "run",
        str(STREAMLIT_APP),

        # Network
        "--server.address",
        STREAMLIT_HOST,

        "--server.port",
        str(STREAMLIT_PORT),

        # Don't open another browser window.
        "--server.headless",
        "true",

        # Disable telemetry.
        "--browser.gatherUsageStats",
        "false",

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The Streamlit app is embedded in an iframe served
        # from FastAPI on port 8000.
        #
        # These are acceptable for this localhost-only setup.
        # Do NOT use these settings for a public deployment.
        # ----------------------------------------------------

        "--server.enableCORS",
        "false",

        "--server.enableXsrfProtection",
        "false",
    ]

    print()
    print("Starting Streamlit...")
    print(
        f"  Internal Streamlit URL:"
        f" http://{STREAMLIT_HOST}:{STREAMLIT_PORT}"
    )

    # --------------------------------------------------------
    # Windows: prevent an additional console window.
    # --------------------------------------------------------

    creationflags = 0

    if sys.platform == "win32":

        creationflags = getattr(
            subprocess,
            "CREATE_NO_WINDOW",
            0,
        )

    # --------------------------------------------------------
    # Start subprocess.
    # --------------------------------------------------------

    streamlit_process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )

    streamlit_started_by_us = True


# ============================================================
# STOP STREAMLIT
# ============================================================

def stop_streamlit() -> None:
    """
    Stop Streamlit only when this FastAPI process started it.
    """

    global streamlit_process
    global streamlit_started_by_us

    if not streamlit_started_by_us:
        return

    if streamlit_process is None:
        return

    if streamlit_process.poll() is not None:

        streamlit_process = None
        streamlit_started_by_us = False

        return

    print("Stopping Streamlit...")

    try:

        streamlit_process.terminate()

        streamlit_process.wait(
            timeout=5
        )

    except subprocess.TimeoutExpired:

        print(
            "Streamlit did not terminate normally. "
            "Forcing shutdown..."
        )

        streamlit_process.kill()

        streamlit_process.wait()

    finally:

        streamlit_process = None
        streamlit_started_by_us = False


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Application startup/shutdown lifecycle.
    """

    # --------------------------------------------------------
    # STARTUP
    # --------------------------------------------------------

    start_streamlit()

    ready = await wait_for_streamlit(
        timeout=20.0
    )

    if not ready:

        print(
            "\nWARNING:"
            "\nStreamlit did not become ready within "
            "the expected time."
        )

    else:

        print()
        print("Streamlit is ready.")

    print()
    print("=" * 60)
    print("Nexa RAG")
    print("=" * 60)
    print(
        f"Frontend : http://{FASTAPI_HOST}:{FASTAPI_PORT}/"
    )
    print(
        f"API Docs : http://{FASTAPI_HOST}:{FASTAPI_PORT}/docs"
    )
    print("=" * 60)
    print()

    yield

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    stop_streamlit()


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Nexa RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

# Existing API routes.
app.include_router(router)


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def root():

    return """
    <!doctype html>

    <html lang="en">

    <head>

        <meta charset="utf-8">

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
        >

        <title>Nexa RAG</title>

        <style>

            html,
            body {
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: #ffffff;
            }

            body {
                font-family:
                    Inter,
                    ui-sans-serif,
                    system-ui,
                    -apple-system,
                    BlinkMacSystemFont,
                    "Segoe UI",
                    sans-serif;
            }

            iframe {
                width: 100%;
                height: 100%;
                border: 0;
                display: block;
            }

        </style>

    </head>

    <body>

        <iframe
            src="http://127.0.0.1:8501/?embed=true&embed_options=hide_loading_screen"
            title="Nexa RAG"
            allow="clipboard-read; clipboard-write"
        ></iframe>

    </body>

    </html>
    """


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=False,
    )