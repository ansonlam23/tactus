"""
Gemini AI Vision helper module.

Uses the Gemini 1.5 Flash REST API directly (no SDK).
Requires internet access and a valid GEMINI_API_KEY in .env.

To activate this backend, set VISION_BACKEND=gemini in .env
and wire it up in main.py alongside azure and local backends.
"""

import base64
import logging
from typing import Optional

import requests

from config import GEMINI_API_KEY, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _call_gemini(image_bytes: bytes, prompt: str) -> str:
    """Send image + prompt to Gemini and return the response text."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Gemini API key missing. Set GEMINI_API_KEY in .env. "
            "Get a free key at https://aistudio.google.com"
        )

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    {"text": prompt},
                ]
            }
        ]
    }

    try:
        response = requests.post(
            _BASE_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Gemini API timed out after {REQUEST_TIMEOUT}s.")
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code
        body = exc.response.text[:300]
        raise RuntimeError(f"Gemini API returned HTTP {status}: {body}")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Gemini API — no internet connection. "
            "Switch to VISION_BACKEND=local for offline use."
        )

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response format: {response.text[:300]}")


# ---------------------------------------------------------------------------
# Public API — mirrors azure_vision.py and local_vision.py interface
# ---------------------------------------------------------------------------


def describe_image(image_bytes: bytes) -> dict:
    """
    Return a natural-language description of the image using Gemini.

    Returns:
        {"caption": str, "tags": list[str], "objects": list[str]}
    """
    logger.info("Calling Gemini Vision API (describe)...")
    caption = _call_gemini(
        image_bytes,
        "Describe what you see in this image in 10 words or fewer.",
    )
    if not caption:
        raise RuntimeError("Gemini returned an empty description.")
    logger.info("Describe result: '%s'", caption[:80])
    return {"caption": caption, "tags": [], "objects": []}


def read_image(image_bytes: bytes) -> dict:
    """
    Extract all text from the image using Gemini OCR.

    Returns:
        {"text": str, "lines": list[str], "guidance": str | None}
    """
    logger.info("Calling Gemini Vision API (read/OCR)...")
    raw = _call_gemini(
        image_bytes,
        "Extract all visible text from this image exactly as it appears. "
        "Return only the text, one line per line, no commentary.",
    )
    if not raw:
        raise RuntimeError("Gemini found no text in this image.")

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    full_text = "\n".join(lines)
    logger.info("Gemini OCR extracted %d lines", len(lines))
    return {"text": full_text, "lines": lines, "guidance": None}
