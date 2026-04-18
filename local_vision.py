"""
Local CV helper module — no internet required.

describe_image(): uses Ollama (vision model, e.g. moondream or llava)
read_image():     uses EasyOCR
"""

import base64
import logging
from typing import Optional

import easyocr
import requests

from config import OLLAMA_MODEL, OLLAMA_URL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# EasyOCR reader is expensive to initialise — do it once at import time.
_ocr_reader: Optional[easyocr.Reader] = None


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initialising EasyOCR reader (first call only)...")
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def describe_image(image_bytes: bytes) -> dict:
    """
    Return a natural-language description of the image using Ollama.

    Returns:
        {"caption": str, "tags": list[str], "objects": list[str]}
    """
    logger.info("Calling Ollama (%s) for image description...", OLLAMA_MODEL)

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Describe what you see in this image in one or two sentences.",
        "images": [b64],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is it running? Run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama timed out after {REQUEST_TIMEOUT}s.")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}")

    caption = response.json().get("response", "").strip()
    if not caption:
        raise RuntimeError("Ollama returned an empty response.")

    logger.info("Describe result: '%s'", caption[:80])
    return {"caption": caption, "tags": [], "objects": []}


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def _check_framing(results: list, image_width: int) -> Optional[str]:
    """Return camera guidance string if framing looks off, otherwise None."""
    if not results or not image_width:
        return None

    cutoff_margin = image_width * 0.01
    center_xs = []

    for (bbox, _text, _conf) in results:
        xs = [pt[0] for pt in bbox]
        min_x, max_x = min(xs), max(xs)
        center_xs.append((min_x + max_x) / 2)

        if min_x < cutoff_margin or max_x > image_width - cutoff_margin:
            return "Text is cut off — back away or reframe the camera."

        if (max_x - min_x) > image_width * 0.75:
            return "Too close — back the camera away."

    if center_xs:
        ratio = (sum(center_xs) / len(center_xs)) / image_width
        if ratio < 0.35:
            return "Text is off to the left — move the camera right."
        if ratio > 0.65:
            return "Text is off to the right — move the camera left."

    return None


def read_image(image_bytes: bytes) -> dict:
    """
    Extract all text from the image using EasyOCR.

    Returns:
        {"text": str, "lines": list[str], "guidance": str | None}
    """
    logger.info("Running EasyOCR on image (%d bytes)...", len(image_bytes))

    reader = _get_ocr_reader()
    results = reader.readtext(image_bytes)

    lines = [text.strip() for (_bbox, text, conf) in results if conf >= 0.3 and text.strip()]

    if not lines:
        raise RuntimeError("EasyOCR found no text in this image.")

    # Estimate image width from the rightmost bbox x-coordinate
    all_xs = [pt[0] for (bbox, _t, _c) in results for pt in bbox]
    image_width = int(max(all_xs)) if all_xs else 0

    guidance = _check_framing(results, image_width)
    if guidance:
        logger.info("Framing guidance: %s", guidance)

    full_text = "\n".join(lines)
    logger.info("EasyOCR extracted %d lines", len(lines))
    return {"text": full_text, "lines": lines, "guidance": guidance}
