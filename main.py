"""
Tactus — FastAPI backend.

Receives images from an ESP32-CAM via multipart/form-data and returns
AI-generated descriptions or OCR text using Azure AI Vision.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from azure_vision import describe_image, read_image
from config import UPLOAD_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

VALID_MODES = {"describe", "read"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(UPLOAD_DIR).mkdir(exist_ok=True)
    logger.info("Server ready. CV backend: Azure AI Vision")
    yield


app = FastAPI(
    title="Tactus",
    description="CV backend for ESP32-CAM accessibility device",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def health_check():
    """Quick health check — call this to confirm the server is running."""
    return {
        "status": "ok",
        "service": "Tactus",
        "upload_dir": UPLOAD_DIR,
    }


@app.post("/process-image")
async def process_image(
    imageFile: UploadFile = File(..., description="Image captured by ESP32-CAM"),
    mode: str = Form(default="describe", description="'describe' or 'read'"),
):
    """
    Process an image and return AI output.

    - **describe**: returns a natural-language caption of the scene
    - **read**: returns OCR text and a list of extracted lines
    """

    # --- Validate mode ---
    if mode not in VALID_MODES:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "mode": mode,
                "filename": None,
                "caption": "",
                "text": "",
                "lines": [],
                "error": f"Invalid mode '{mode}'. Must be 'describe' or 'read'.",
            },
        )

    # --- Read image bytes ---
    image_bytes = await imageFile.read()
    if not image_bytes:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "mode": mode,
                "filename": None,
                "caption": "",
                "text": "",
                "lines": [],
                "error": "Uploaded file is empty.",
            },
        )

    logger.info(
        "Received image: '%s' (%d bytes) | mode: %s",
        imageFile.filename,
        len(image_bytes),
        mode,
    )

    # --- Save debug copy ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    original_stem = Path(imageFile.filename or "image").stem
    save_filename = f"{timestamp}_{uid}_{original_stem}.jpg"
    save_path = Path(UPLOAD_DIR) / save_filename

    try:
        save_path.write_bytes(image_bytes)
        logger.info("Saved debug copy: %s", save_path)
    except OSError as exc:
        # Non-fatal — log and continue processing
        logger.warning("Could not save debug image: %s", exc)

    # --- Run computer vision ---
    try:
        if mode == "describe":
            result = describe_image(image_bytes)
            return {
                "success": True,
                "mode": mode,
                "filename": save_filename,
                "caption": result.get("caption", ""),
                "text": result.get("caption", ""),  # convenience alias
                "lines": [],
                "error": None,
            }

        else:  # mode == "read"
            result = read_image(image_bytes)
            return {
                "success": True,
                "mode": mode,
                "filename": save_filename,
                "caption": "",
                "text": result.get("text", ""),
                "lines": result.get("lines", []),
                "guidance": result.get("guidance"),
                "error": None,
            }

    except TimeoutError as exc:
        logger.error("CV timeout: %s", exc)
        return JSONResponse(
            status_code=504,
            content=_error_response(mode, save_filename, f"Timeout: {exc}"),
        )
    except Exception as exc:
        logger.exception("CV processing failed")
        return JSONResponse(
            status_code=502,
            content=_error_response(mode, save_filename, str(exc)),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_response(mode: str, filename: str, error: str) -> dict:
    return {
        "success": False,
        "mode": mode,
        "filename": filename,
        "caption": "",
        "text": "",
        "lines": [],
        "error": error,
    }
