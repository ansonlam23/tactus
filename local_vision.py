"""
Local CV helper module — no internet required.

describe_image(): uses Ollama (vision model, e.g. moondream or llava)
read_image():     uses EasyOCR
"""

import base64
import logging
from typing import Optional

import pytesseract
import requests
from PIL import Image
import io

from config import OLLAMA_MODEL, OLLAMA_URL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


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
        "prompt": "Describe this image in 10 words or fewer.",
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


def _check_framing(image: Image.Image) -> Optional[str]:
    """Return camera guidance string based on Tesseract bounding boxes, or None."""
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return None

    img_w = image.size[0]
    if not img_w:
        return None

    cutoff_margin = img_w * 0.01
    center_xs = []

    for i, conf in enumerate(data["conf"]):
        try:
            if int(conf) < 30:
                continue
        except (ValueError, TypeError):
            continue

        left  = data["left"][i]
        width = data["width"][i]
        right = left + width

        if width == 0:
            continue

        center_xs.append(left + width / 2)

        if left < cutoff_margin or right > img_w - cutoff_margin:
            return "Text is cut off — back away or reframe the camera."

        if width > img_w * 0.75:
            return "Too close — back the camera away."

    if center_xs:
        ratio = (sum(center_xs) / len(center_xs)) / img_w
        if ratio < 0.35:
            return "Text is off to the left — move the camera right."
        if ratio > 0.65:
            return "Text is off to the right — move the camera left."

    return None


def read_image(image_bytes: bytes) -> dict:
    """
    Extract all text from the image using Tesseract OCR.

    Returns:
        {"text": str, "lines": list[str], "guidance": str | None}
    """
    logger.info("Running Tesseract OCR on image (%d bytes)...", len(image_bytes))

    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Auto-detect and correct rotation
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if angle:
            image = image.rotate(angle, expand=True)
    except Exception:
        pass

    raw_text = pytesseract.image_to_string(image, config="--psm 3")

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    if not lines:
        raise RuntimeError("Tesseract found no text in this image.")

    full_text = "\n".join(lines)
    guidance = _check_framing(image)
    if guidance:
        logger.info("Framing guidance: %s", guidance)
    logger.info("Tesseract extracted %d lines", len(lines))
    return {"text": full_text, "lines": lines, "guidance": guidance}
