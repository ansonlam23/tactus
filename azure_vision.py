"""
Azure AI Vision helper module.

Uses the Image Analysis v4.0 REST API directly (no SDK) for reliability.
All calls are synchronous — no polling required with the v4.0 endpoint.
"""

import logging

import requests

from config import REQUEST_TIMEOUT, VISION_ENDPOINT, VISION_KEY

logger = logging.getLogger(__name__)

_API_VERSION = "2024-02-01"
_ANALYZE_PATH = "/computervision/imageanalysis:analyze"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_url() -> str:
    return f"{VISION_ENDPOINT}{_ANALYZE_PATH}"


def _headers() -> dict[str, str]:
    return {
        "Ocp-Apim-Subscription-Key": VISION_KEY,
        "Content-Type": "application/octet-stream",
    }


def _call_azure(image_bytes: bytes, features: str) -> dict:
    """POST raw image bytes to Azure Image Analysis v4.0 and return parsed JSON."""
    if not VISION_ENDPOINT or not VISION_KEY:
        raise RuntimeError(
            "Azure credentials missing. Set VISION_ENDPOINT and VISION_KEY in .env."
        )

    params = {
        "api-version": _API_VERSION,
        "features": features,
        "language": "en",
    }

    try:
        response = requests.post(
            _build_url(),
            headers=_headers(),
            params=params,
            data=image_bytes,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"Azure Vision API timed out after {REQUEST_TIMEOUT}s. "
            "Check your network or increase REQUEST_TIMEOUT in .env."
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code
        body = exc.response.text[:300]
        raise RuntimeError(f"Azure Vision API returned HTTP {status}: {body}")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error calling Azure Vision: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def describe_image(image_bytes: bytes) -> dict:
    """
    Return a short natural-language caption for the image.

    Returns:
        {"caption": str, "confidence": float}
    """
    logger.info("Calling Azure Vision API (feature: caption)...")
    data = _call_azure(image_bytes, features="caption")

    caption_result = data.get("captionResult", {})
    caption = caption_result.get("text", "")
    confidence = round(caption_result.get("confidence", 0.0), 4)

    if not caption:
        raise RuntimeError("Azure returned no caption for this image.")

    logger.info("Caption: '%s' (confidence: %.2f)", caption, confidence)
    return {"caption": caption, "confidence": confidence}


def read_image(image_bytes: bytes) -> dict:
    """
    Extract all text from the image using OCR.

    Returns:
        {"text": str, "lines": list[str]}
    """
    logger.info("Calling Azure Vision API (feature: read)...")
    data = _call_azure(image_bytes, features="read")

    read_result = data.get("readResult", {})
    full_text: str = read_result.get("content", "")

    lines: list[str] = []
    for page in read_result.get("pages", []):
        for line in page.get("lines", []):
            content = line.get("content", "").strip()
            if content:
                lines.append(content)

    if not full_text and lines:
        full_text = "\n".join(lines)

    if not full_text:
        raise RuntimeError("Azure returned no text for this image.")

    logger.info("OCR extracted %d lines", len(lines))
    return {"text": full_text, "lines": lines}
