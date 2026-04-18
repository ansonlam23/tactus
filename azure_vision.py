"""
Azure AI Vision helper module.

Uses the Image Analysis v4.0 REST API directly (no SDK) for reliability.
All calls are synchronous — no polling required with the v4.0 endpoint.
"""

import logging
from typing import Optional

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
    Return tags and detected objects for the image.

    Returns:
        {"caption": str, "tags": list[str], "objects": list[str]}
    """
    logger.info("Calling Azure Vision API (features: tags,objects)...")
    data = _call_azure(image_bytes, features="tags,objects")

    tags = [
        t["name"]
        for t in data.get("tagsResult", {}).get("values", [])
        if t.get("confidence", 0) >= 0.7
    ]

    objects = [
        o["tags"][0]["name"]
        for o in data.get("objectsResult", {}).get("values", [])
        if o.get("tags") and o["tags"][0].get("confidence", 0) >= 0.7
    ]

    combined = list(dict.fromkeys(objects + tags))  # objects first, deduped
    caption = ", ".join(combined) if combined else ""

    if not caption:
        raise RuntimeError("Azure returned no tags or objects for this image.")

    logger.info("Describe result: '%s'", caption)
    return {"caption": caption, "tags": tags, "objects": objects}


def _check_framing(data: dict) -> Optional[str]:
    """Return a camera guidance string if framing is off, otherwise None."""
    width = data.get("metadata", {}).get("width", 0)
    if not width:
        return None

    all_lines = []
    for block in data.get("readResult", {}).get("blocks", []):
        all_lines.extend(block.get("lines", []))

    if not all_lines:
        return None

    cutoff_margin = width * 0.01  # truly touching the edge
    center_xs = []

    for line in all_lines:
        poly = line.get("boundingPolygon", [])
        if not poly:
            continue
        xs = [p["x"] for p in poly]
        min_x, max_x = min(xs), max(xs)
        center_xs.append((min_x + max_x) / 2)

        if min_x < cutoff_margin or max_x > width - cutoff_margin:
            return "Text is cut off — back away or reframe the camera."

        if (max_x - min_x) > width * 0.75:
            return "Too close — back the camera away."

    if center_xs:
        ratio = (sum(center_xs) / len(center_xs)) / width
        if ratio < 0.35:
            return "Text is off to the left — move the camera right."
        if ratio > 0.65:
            return "Text is off to the right — move the camera left."

    return None


def read_image(image_bytes: bytes) -> dict:
    """
    Extract all text from the image using OCR.

    Returns:
        {"text": str, "lines": list[str], "guidance": str | None}
    """
    logger.info("Calling Azure Vision API (feature: read)...")
    data = _call_azure(image_bytes, features="read")

    read_result = data.get("readResult", {})

    lines: list[str] = []
    for block in read_result.get("blocks", []):
        for line in block.get("lines", []):
            content = line.get("text", "").strip()
            if content:
                lines.append(content)

    full_text = "\n".join(lines)

    if not full_text:
        raise RuntimeError("Azure returned no text for this image.")

    guidance = _check_framing(data)
    if guidance:
        logger.info("Framing guidance: %s", guidance)

    logger.info("OCR extracted %d lines", len(lines))
    return {"text": full_text, "lines": lines, "guidance": guidance}
